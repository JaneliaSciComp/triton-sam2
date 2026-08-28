import io

import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils

# Per-model routing table. Adding a new SAM family is one entry here:
#   - expected_size: the decoded JPEG must already be this size (this model does
#     not resize; the client is responsible for resizing/padding before encoding)
#   - encoder_model: the DEPLOYED encoder to dispatch to via BLS
#   - encoder_input: that encoder's input tensor name
#   - encoder_outputs: the tensors it returns (also the outputs this model emits
#     for that model_type; the caller must request exactly these)
ROUTES = {
    "sam1": {
        "expected_size": (1024, 1024),
        "encoder_model": "sam1_encoder",
        "encoder_input": "image",
        "encoder_outputs": ["image_embeddings"],
    },
    "sam2": {
        "expected_size": (1024, 1024),
        "encoder_model": "sam2.1_large_encoder",
        "encoder_input": "image",
        "encoder_outputs": ["high_res_feats_0", "high_res_feats_1", "image_embed"],
    },
    "sam3": {
        "expected_size": (1008, 1008),
        "encoder_model": "sam3_encoder",
        "encoder_input": "pixel_values",
        "encoder_outputs": ["image_embeddings.0", "image_embeddings.1", "image_embeddings.2"],
    },
}


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            # A gRPC client may have cancelled this request while it was queued
            # behind earlier work; skip the JPEG decode and the expensive BLS
            # encoder call instead of computing a result nobody will read. This
            # is the only cancellation point: the default (non-batching)
            # scheduler never drops cancelled requests itself, it just discards
            # their responses after execution.
            if request.is_cancelled():
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("request was cancelled", pb_utils.TritonError.CANCELLED)
                    )
                )
                continue

            # "model_type" is a BYTES tensor of shape [1]; pick the route
            model_type_raw = pb_utils.get_input_tensor_by_name(request, "model_type").as_numpy()[0]
            model_type = (
                model_type_raw.decode("utf-8") if isinstance(model_type_raw, bytes) else str(model_type_raw)
            ).strip().lower()

            route = ROUTES.get(model_type)
            if route is None:
                raise pb_utils.TritonModelException(
                    f"unknown model_type {model_type!r}; expected one of {sorted(ROUTES)}"
                )

            # "encoded_image" is a BYTES tensor of shape [1]; one JPEG per request
            encoded_image = pb_utils.get_input_tensor_by_name(request, "encoded_image").as_numpy()[0]

            # decode JPEG -> HWC uint8 RGB (assumed already at the expected size)
            decoded_image = Image.open(io.BytesIO(encoded_image)).convert("RGB")

            # cheap guard: this model does not resize, so a wrong-size JPEG would
            # silently produce a wrong-shape tensor the encoder rejects
            expected_size = route["expected_size"]
            if decoded_image.size != expected_size:
                raise pb_utils.TritonModelException(
                    f"model_type {model_type!r} expects a {expected_size} JPEG, but the image "
                    f"decoded to {decoded_image.size}; the client must resize before encoding "
                    "(this model does not resize)"
                )

            # build the encoder's tensor: [1,3,H,W] FP32, planar RGB, [0,1]
            normalized_hwc = np.asarray(decoded_image, dtype=np.float32) / 255.0  # [H,W,3]
            planar_chw = np.transpose(normalized_hwc, (2, 0, 1))                  # [3,H,W]
            batched_chw = np.ascontiguousarray(planar_chw[None])                  # [1,3,H,W]

            # dispatch to the chosen encoder via BLS and collect its outputs
            encoder_request = pb_utils.InferenceRequest(
                model_name=route["encoder_model"],
                requested_output_names=route["encoder_outputs"],
                inputs=[pb_utils.Tensor(route["encoder_input"], batched_chw)],
            )
            encoder_response = encoder_request.exec()
            if encoder_response.has_error():
                raise pb_utils.TritonModelException(
                    f"{route['encoder_model']} inference failed: {encoder_response.error().message()}"
                )

            # Pass the encoder's output tensors straight through under the same
            # names. We reuse the tensor objects rather than calling .as_numpy():
            # the encoder runs on GPU and returns GPU-resident tensors, which
            # .as_numpy() cannot convert ("Tensor is stored in GPU"). Handing the
            # tensors back directly lets Triton move them to the client itself.
            output_tensors = [
                pb_utils.get_output_tensor_by_name(encoder_response, output_name)
                for output_name in route["encoder_outputs"]
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses
