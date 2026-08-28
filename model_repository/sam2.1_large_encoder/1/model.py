import triton_python_backend_utils as pb_utils

INNER_MODEL = "sam2.1_large_encoder_onnx"
OUTPUT_NAMES = ["high_res_feats_0", "high_res_feats_1", "image_embed"]


class TritonPythonModel:
    """Cancel-aware passthrough for the SAM2.1 large encoder.

    The dynamic-batch scheduler (see config.pbtxt) already drops queued
    requests that were cancelled by the client before they reach execute();
    the is_cancelled() check below additionally catches a cancellation that
    lands after scheduling but before the BLS dispatch."""

    def execute(self, requests):
        responses = []
        for request in requests:
            if request.is_cancelled():
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("request was cancelled", pb_utils.TritonError.CANCELLED)
                    )
                )
                continue

            image = pb_utils.get_input_tensor_by_name(request, "image")
            inner_response = pb_utils.InferenceRequest(
                model_name=INNER_MODEL,
                requested_output_names=OUTPUT_NAMES,
                inputs=[image],
            ).exec()
            if inner_response.has_error():
                raise pb_utils.TritonModelException(
                    f"{INNER_MODEL} inference failed: {inner_response.error().message()}"
                )

            # Hand the (possibly GPU-resident) tensors back untouched; calling
            # .as_numpy() on GPU tensors would fail and force a host copy.
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.get_output_tensor_by_name(inner_response, name)
                        for name in OUTPUT_NAMES
                    ]
                )
            )
        return responses
