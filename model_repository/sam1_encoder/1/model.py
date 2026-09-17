"""Cancel-aware encoder front end accepting either a raw tensor or a JPEG.

One file, three deployments: this exact module is used by `sam1_encoder`,
`sam2.1_large_encoder` and `sam3_tracker_encoder_fp16`. Everything
model-specific comes from that model's `config.pbtxt` `parameters` block, so the
copies must stay byte-identical -- edit one and copy it to the other two.

Half-precision responses: a client may send the request parameter
`request_fp16: true` to get the embeddings cast to FLOAT16, halving the
response (SAM2: 16.8 MB -> 8.4 MB). Omitting it returns FP32, so clients that
predate the option are unaffected -- which is why the option is opt-in rather
than a separate endpoint. For SAM2 the cast is provably lossless relative to
the old `sam2.1_large_encoder_fp16` model: that export was the same fp32
network with three Cast->FLOAT16 nodes bolted on its outputs, and
`fp32_output.astype(float16)` was verified bit-identical to what it returned.
Casting here rather than running a second copy of the weights also halves the
device-to-host transfer, so the fp16 path is cheaper, not more expensive.

Why a Python model owns each public encoder name:

  * The ONNX backend cannot have an optional input and cannot decode a JPEG, so
    "one endpoint, either input" has to be implemented in front of it. The raw
    ONNX models live under `<name>_onnx` and are reached from here via BLS.
  * The encoder exports hard-wire batch=1, so they cannot run under the
    dynamic-batch scheduler -- and that scheduler is what drops CANCELLED
    requests before they execute. A Python model has no baked batch dimension,
    so it can batch trivially at size 1 and the cancellation/priority behaviour
    comes back.

Normalization is applied ONLY on the JPEG path. The raw tensor input is passed
through untouched: it is "the exact tensor for the target model", already
normalized by the caller. A JPEG only carries uint8, so for that path the server
owns the affine step -- none of the encoders normalize internally (each goes
straight from its input into the patch-embed Conv).
"""

import io
import json

import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils

# Second, optional input. Same name on every endpoint so a client does not have
# to special-case the model family for the JPEG path (the raw tensor input name
# does differ: `image` for SAM1/SAM2, `pixel_values` for SAM3).
JPEG_INPUT = "jpeg_image"

# Request parameter (not an input tensor) selecting a half-precision response.
FP16_PARAM = "request_fp16"


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        parameters = {
            key: value["string_value"]
            for key, value in model_config.get("parameters", {}).items()
        }

        # The raw ONNX model this front end forwards to, and its input tensor
        # name -- which is deliberately also OUR input tensor name, so the raw
        # path can hand the client's tensor straight over without renaming it.
        self.inner_model = parameters["inner_model"]
        self.tensor_input = parameters["tensor_input"]
        self.output_names = [name.strip() for name in parameters["output_names"].split(",")]

        # This model does not resize. A JPEG must already be square at the
        # encoder's input size; anything else is a client bug we would otherwise
        # turn into a confusing shape error from the ONNX model.
        edge = int(parameters["expected_size"])
        self.expected_size = (edge, edge)

        # normalized = ((x / 255) - mean) / std, per channel, RGB order.
        # SAM1/SAM2 use the torchvision ImageNet constants (identical to SAM's
        # canonical pixel_mean/pixel_std of [123.675, 116.28, 103.53] /
        # [58.395, 57.12, 57.375] once those are divided by 255); SAM3 uses
        # 0.5/0.5, which just maps to [-1, 1].
        self.norm_mean = np.asarray(
            [float(v) for v in parameters["norm_mean"].split(",")], dtype=np.float32
        )
        self.norm_std = np.asarray(
            [float(v) for v in parameters["norm_std"].split(",")], dtype=np.float32
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            raw_tensor = pb_utils.get_input_tensor_by_name(request, self.tensor_input)
            jpeg_tensor = pb_utils.get_input_tensor_by_name(request, JPEG_INPUT)

            # Exactly one of the two. Checked before anything else so a
            # malformed request fails on its own terms rather than being
            # reported as a cancellation or as an error from the inner model.
            if (raw_tensor is None) == (jpeg_tensor is None):
                supplied = "both were" if raw_tensor is not None else "neither was"
                responses.append(
                    self._error(
                        f"supply exactly one of {self.tensor_input!r} or "
                        f"{JPEG_INPUT!r} ({supplied} supplied)"
                    )
                )
                continue

            # A gRPC client may have cancelled this while it was queued behind
            # earlier work. The dynamic-batch scheduler drops most of those
            # before we are called; this catches one that landed after
            # scheduling, and skips both the decode and the expensive encode.
            if request.is_cancelled():
                responses.append(
                    self._error("request was cancelled", pb_utils.TritonError.CANCELLED)
                )
                continue

            if jpeg_tensor is None:
                # Raw path: pass the client's tensor through untouched. Note we
                # reuse the Tensor object rather than calling .as_numpy() -- it
                # may already be GPU-resident, and a host round trip would cost
                # 12.6 MB per request for nothing.
                encoder_inputs = [raw_tensor]
            else:
                try:
                    decoded = self._decode(jpeg_tensor)
                except ValueError as exc:
                    responses.append(self._error(str(exc)))
                    continue
                encoder_inputs = [pb_utils.Tensor(self.tensor_input, decoded)]

            # Decide this before dispatching: when a cast is coming, ask the
            # encoder to hand its outputs back in HOST memory. That copy has to
            # happen anyway for the response to reach the client, and doing it
            # here keeps the cast device-agnostic. Casting on the GPU instead
            # would have to follow the inner model's card -- cupy defaults to
            # device 0, while e.g. sam1_encoder_onnx is pinned to card 1, which
            # fails with "Failed to synchronize CUDA device with id 0".
            want_fp16 = self._wants_fp16(request)
            inner_request = pb_utils.InferenceRequest(
                model_name=self.inner_model,
                requested_output_names=self.output_names,
                inputs=encoder_inputs,
                **(
                    {
                        "preferred_memory": pb_utils.PreferredMemory(
                            pb_utils.TRITONSERVER_MEMORY_CPU, 0
                        )
                    }
                    if want_fp16
                    else {}
                ),
            )
            inner_response = inner_request.exec()
            if inner_response.has_error():
                responses.append(
                    self._error(
                        f"{self.inner_model} inference failed: "
                        f"{inner_response.error().message()}"
                    )
                )
                continue

            tensors = [
                pb_utils.get_output_tensor_by_name(inner_response, name)
                for name in self.output_names
            ]
            if want_fp16:
                try:
                    tensors = [self._to_fp16(t) for t in tensors]
                except Exception as exc:
                    responses.append(
                        self._error(f"could not cast outputs to fp16: {exc}")
                    )
                    continue

            # Otherwise hand the encoder's tensors back untouched and under the
            # same names, without .as_numpy(): they are GPU-resident, and Triton
            # can move them to the client itself.
            responses.append(pb_utils.InferenceResponse(output_tensors=tensors))
        return responses

    @staticmethod
    def _wants_fp16(request):
        """True if the request asked for a half-precision response.

        Absent parameter -> False, so a client that predates the option keeps
        getting FP32. Accepts a real JSON boolean or the string "true", since
        HTTP clients often stringify parameters.
        """
        try:
            raw = request.parameters()
        except Exception:
            return False
        if not raw:
            return False
        try:
            value = json.loads(raw).get(FP16_PARAM, False)
        except (ValueError, AttributeError):
            return False
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    @staticmethod
    def _to_fp16(tensor):
        """Cast an output tensor to FLOAT16.

        The tensor is CPU-resident here because the BLS request asked for host
        memory whenever a cast was coming. numpy rounds to nearest even, which
        is what an ONNX Cast does, so the result is bit-identical to running a
        model that had the Cast baked into its graph -- verified against the
        retired sam2.1_large_encoder_fp16 export.

        Triton does not enforce the dtype declared in config.pbtxt for python
        models (verified on 25.01), which is what lets one output name carry
        either precision. The declared FP32 is a default, not a guarantee; see
        the note in config.pbtxt.
        """
        return pb_utils.Tensor(tensor.name(), tensor.as_numpy().astype(np.float16))

    def _decode(self, jpeg_tensor):
        """JPEG bytes -> the encoder's [1, 3, H, W] FP32 input tensor."""
        # reshape(-1)[0] so this works whether the input arrives as [1] or, under
        # the batching scheduler, as [1, 1].
        encoded = jpeg_tensor.as_numpy().reshape(-1)[0]

        # RGB matters: the mean/std are per-channel and differ from each other,
        # so a BGR decode would normalize the red and blue channels wrongly.
        decoded = Image.open(io.BytesIO(encoded)).convert("RGB")
        if decoded.size != self.expected_size:
            raise ValueError(
                f"{JPEG_INPUT} decoded to {decoded.size}, expected "
                f"{self.expected_size}; the client must resize before encoding "
                "(this model does not resize)"
            )

        pixels = np.asarray(decoded, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
        pixels = (pixels - self.norm_mean) / self.norm_std      # broadcast over channels
        planar = np.transpose(pixels, (2, 0, 1))                # [3, H, W]
        return np.ascontiguousarray(planar[None])               # [1, 3, H, W]

    @staticmethod
    def _error(message, code=None):
        error = (
            pb_utils.TritonError(message, code)
            if code is not None
            else pb_utils.TritonError(message)
        )
        return pb_utils.InferenceResponse(error=error)
