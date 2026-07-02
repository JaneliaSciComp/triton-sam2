import io

import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils

# The SAM2 encoder expects a fixed-size square image. The client is responsible
# for resizing/padding to this size before JPEG-encoding; this model only
# decompresses, it does not resize.
EXPECTED_SIZE = (1024, 1024)


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            # "encoded_image" is a BYTES tensor of shape [1]; one JPEG per request
            encoded_image = pb_utils.get_input_tensor_by_name(request, "encoded_image").as_numpy()[0]

            # decode JPEG -> HWC uint8 RGB (assumed already 1024x1024)
            decoded_image = Image.open(io.BytesIO(encoded_image)).convert("RGB")

            # cheap guard: this minimal model does not resize, so a wrong-size
            # JPEG would silently produce a wrong-shape tensor the encoder rejects
            if decoded_image.size != EXPECTED_SIZE:
                raise pb_utils.TritonModelException(
                    f"encoded_image decoded to {decoded_image.size}, expected {EXPECTED_SIZE}; "
                    "the client must resize before encoding (this model does not resize)"
                )

            # build the encoder's tensor: [1,3,1024,1024] FP32, planar RGB, [0,1]
            normalized_hwc = np.asarray(decoded_image, dtype=np.float32) / 255.0  # [H,W,3]
            planar_chw = np.transpose(normalized_hwc, (2, 0, 1))                  # [3,H,W]
            batched_chw = np.ascontiguousarray(planar_chw[None])                  # [1,3,H,W]

            output_tensor = pb_utils.Tensor("image", batched_chw)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses
