# `sam_encoder_jpeg` — gRPC client guide

A single Triton endpoint that turns a JPEG into SAM encoder embeddings. The
client sends **one JPEG plus a model-type flag**; the server decodes the image,
runs the matching SAM encoder, and returns its embeddings — in one call. This
replaces sending a large raw FP32 tensor over the wire (~12.6 MB for 1024×1024).

## Endpoint

| Env  | gRPC address                             | TLS                     |
|------|------------------------------------------|-------------------------|
| dev  | `sam-inference-dev.int.janelia.org:443`  | yes (passthrough)       |
| prod | `sam-inference.janelia.org:443`          | yes (passthrough)       |

- Protocol: Triton / KServe v2 **gRPC** (`ModelInfer`).
- TLS is terminated at the server, so connect with TLS enabled (`ssl=True`). If
  your platform doesn't trust the server's CA, supply the CA cert explicitly
  (see troubleshooting).
- Model name: **`sam_encoder_jpeg`**

## Request contract

**Inputs** (both are `BYTES` tensors of shape `[1]`):

| name            | type    | value                                             |
|-----------------|---------|---------------------------------------------------|
| `encoded_image` | `BYTES` | one JPEG, already sized to the model input (below) |
| `model_type`    | `BYTES` | `"sam1"`, `"sam2"`, or `"sam3"`                    |

**Outputs** — depend on `model_type`. **You must explicitly request the output
names for your model** (a request with no outputs specified asks Triton for *all*
declared outputs and will fail, since only one model's tensors are produced per
call):

| `model_type` | JPEG size   | request these output names                                |
|--------------|-------------|-----------------------------------------------------------|
| `sam1`       | 1024×1024   | `image_embeddings`                                        |
| `sam2`       | 1024×1024   | `high_res_feats_0`, `high_res_feats_1`, `image_embed`     |
| `sam3`       | 1008×1008   | `image_embeddings.0`, `image_embeddings.1`, `image_embeddings.2` |

Output shapes (FP32):
- sam1: `image_embeddings` `[1,256,64,64]`
- sam2: `high_res_feats_0` `[1,32,256,256]`, `high_res_feats_1` `[1,64,128,128]`, `image_embed` `[1,256,64,64]`
- sam3: three multi-scale embeddings (dynamic shapes)

These are identical to what the encoders return today, so your downstream
decoder stage is unchanged.

## Two rules that will bite if you get them wrong

1. **Send a correctly-sized JPEG.** The server does **not** resize — it only
   JPEG-decodes and scales to `[0,1]`. The decoded image must be exactly
   1024×1024 (sam1/sam2) or 1008×1008 (sam3), or the request is rejected. Do the
   same resize/pad you already do to build the current FP32 tensor, then
   JPEG-encode those pixels instead of sending raw floats.
2. **The JPEG must be RGB.** The server decodes with `PIL.Image.convert("RGB")`.
   A standard JPEG written from an OpenCV BGR image decodes correctly (OpenCV
   handles the channel order on encode) — see the example.

## Python example (`tritonclient[grpc]`)

```python
import cv2
import numpy as np
import tritonclient.grpc as grpcclient

HOST = "sam-inference-dev.int.janelia.org:443"   # prod: sam-inference.janelia.org:443
MODEL = "sam_encoder_jpeg"
MODEL_TYPE = "sam2"                               # "sam1" | "sam2" | "sam3"

SIZE = 1008 if MODEL_TYPE == "sam3" else 1024
OUTPUTS = {
    "sam1": ["image_embeddings"],
    "sam2": ["high_res_feats_0", "high_res_feats_1", "image_embed"],
    "sam3": ["image_embeddings.0", "image_embeddings.1", "image_embeddings.2"],
}[MODEL_TYPE]

# 1) Produce the pixels you already feed the encoder, at the model's square input
#    size, then JPEG-encode them. Replace the resize with your existing
#    resize/pad logic if it differs.
image_bgr = cv2.imread("cell.png")               # BGR uint8
image_bgr = cv2.resize(image_bgr, (SIZE, SIZE))
ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
jpeg_bytes = buffer.tobytes()

# 2) Build the two BYTES inputs.
encoded_image = grpcclient.InferInput("encoded_image", [1], "BYTES")
encoded_image.set_data_from_numpy(np.array([jpeg_bytes], dtype=object))
model_type = grpcclient.InferInput("model_type", [1], "BYTES")
model_type.set_data_from_numpy(np.array([MODEL_TYPE.encode()], dtype=object))

# 3) Infer, requesting only this model's outputs.
client = grpcclient.InferenceServerClient(url=HOST, ssl=True)
response = client.infer(
    MODEL,
    inputs=[encoded_image, model_type],
    outputs=[grpcclient.InferRequestedOutput(name) for name in OUTPUTS],
)

for name in OUTPUTS:
    embedding = response.as_numpy(name)
    print(name, embedding.shape, embedding.dtype)
```

Install: `pip install "tritonclient[grpc]" numpy opencv-python-headless`

## Other languages (Kotlin / Java / Swift)

The contract is language-agnostic — any Triton/KServe v2 gRPC client works:

- Use the KServe v2 protos (`grpc_service.proto`, `model_config.proto`) shipped
  with Triton and call the `ModelInfer` RPC on model `sam_encoder_jpeg`.
- Set the two inputs as `BYTES` (`datatype: "BYTES"`, `shape: [1]`). In the raw
  gRPC contents, `BYTES` tensors go in `raw_input_contents` as
  length-prefixed elements: a little-endian `uint32` byte count followed by the
  bytes — one such element per input (the JPEG for `encoded_image`, the UTF-8
  string e.g. `"sam2"` for `model_type`). Most Triton client libraries do this
  framing for you.
- Populate `outputs` with the names from the table above.
- Connect with TLS on port 443.

## Troubleshooting

- **`unknown model_type '...'`** — `model_type` must be exactly `sam1`, `sam2`,
  or `sam3` (lowercase).
- **`expects a (H, W) JPEG, but the image decoded to (h, w)`** — your JPEG isn't
  the required size; resize before encoding.
- **A request that returns "output '...' is not produced"** — you requested an
  output name that doesn't belong to your `model_type`, or you didn't specify
  outputs at all. Request exactly the names in the table.
- **TLS/certificate verification error** — pass your CA to the client, e.g.
  `grpcclient.InferenceServerClient(url=HOST, ssl=True, root_certificates="ca.pem")`.
- **Quick reachability check** (no inference):
  ```python
  c = grpcclient.InferenceServerClient(url=HOST, ssl=True)
  print(c.is_server_ready(), c.is_model_ready("sam_encoder_jpeg"))
  ```

## Notes

- One JPEG per request (`encoded_image` shape is `[1]`); no batching.
- As of now the **dev** server only has the SAM2 encoder loaded, so only
  `model_type="sam2"` returns embeddings there; `sam1`/`sam3` need their encoders
  present on that server.
