# Encoder endpoints: gRPC client guide

Every encoder endpoint takes **either** a raw FP32 tensor **or** a single JPEG.
Sending a JPEG saves ~12.6 MB of upload per 1024×1024 image.

Supersedes the old `sam_encoder_jpeg` guide. That endpoint, along with
`sam2_encoder_jpeg`, has been removed — JPEG support is now an optional input on
each encoder's own name, so there is no separate JPEG endpoint and no
`model_type` flag.

## Endpoints

| endpoint | raw input | shape | JPEG size | outputs | output dtype |
|---|---|---|---|---|---|
| `sam1_encoder` | `image` | 1×3×1024×1024 | 1024×1024 | `image_embeddings` | FP32 |
| `sam2.1_large_encoder` | `image` | 1×3×1024×1024 | 1024×1024 | `image_embed`<br>`high_res_feats_0`<br>`high_res_feats_1` | FP32 |
| `sam2.1_large_encoder_fp16` | `image` | 1×3×1024×1024 | 1024×1024 | same three | **FP16** |
| `sam3_tracker_encoder_fp16` | `pixel_values` | 1×3×1008×1008 | 1008×1008 | `image_embeddings.0`<br>`image_embeddings.1`<br>`image_embeddings.2` | FP32 |

`sam2.1_large_encoder` is FP32 and stays FP32 — released Paintera builds depend
on it. `sam2.1_large_encoder_fp16` is the same model with the response halved
(16.8 MB → 8.4 MB); its *input* is still FP32.

SAM3 differs from the others in input name **and** size: `pixel_values` at
1008×1008. The `_fp16` in its name refers to its weights, not its response.

## The two inputs

| input | type | shape | meaning |
|---|---|---|---|
| `image` / `pixel_values` | `FP32` | the endpoint's raw shape | the exact tensor for the model, passed straight through |
| `jpeg_image` | `BYTES` | `[1, 1]` | one JPEG, already scaled to the endpoint's size |

Both are optional; supply **exactly one**. Supplying both, or neither, returns
an error naming the two inputs.

> **`jpeg_image` is `[1, 1]`, not `[1]`.** These endpoints run under Triton's
> batching scheduler (that is what lets a cancelled request be dropped from the
> queue instead of being encoded anyway), and under it the leading batch
> dimension is implicit in the config but still present on the wire. So a
> one-element BYTES tensor is shaped `[1, 1]`. The raw `image` shape is
> unaffected — it stays `1×3×1024×1024` exactly as before.

Outputs may be requested by name or left unspecified — each endpoint produces
only its own tensors, so "all" is now correct (unlike the old union-output
`sam_encoder_jpeg`).

## Preparing a JPEG

The server does **not** resize. Send a square JPEG at exactly the endpoint's
size or you get an error naming the decoded size.

The server **does** normalize, because none of the encoder graphs normalize
internally. After decoding to RGB it applies, per channel:

```
normalized = ((x / 255) - mean) / std
```

| endpoint | mean | std |
|---|---|---|
| `sam1_encoder` | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
| `sam2.1_large_encoder`, `..._fp16` | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
| `sam3_tracker_encoder_fp16` | 0.5, 0.5, 0.5 | 0.5, 0.5, 0.5 |

SAM1/SAM2 use the torchvision ImageNet constants — identical to SAM's canonical
`pixel_mean`/`pixel_std` of `[123.675, 116.28, 103.53]` / `[58.395, 57.12,
57.375]` divided by 255. SAM3 is not ImageNet; it just maps to [-1, 1].

**Do this yourself on the raw `image` path.** That input is passed through
untouched, so a client sending un-normalized values gets un-normalized encoder
input. The two paths only agree if the client applies the same constants the
table above lists.

Two caveats worth knowing:

- **JPEG is lossy**, so the two paths will never be bit-identical for the same
  image — chroma subsampling in particular shifts red and blue slightly. Use
  high quality, and expect cosine similarity ~1.0 rather than exact equality.
- **Padding.** Upstream SAM2 and SAM3 resize to a square and never pad. Upstream
  SAM1 pads with zeros *after* normalizing, which a pre-padded JPEG cannot
  reproduce: a black border in the JPEG normalizes to ≈ -2.1, not 0. If you
  letterbox non-square images for `sam1_encoder`, ask about adding an optional
  `valid_size` input rather than relying on the border being neutral.

## Example (Python, gRPC)

```python
import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient("sam-inference.janelia.org:443", ssl=True)

# --- JPEG path -------------------------------------------------------------
with open("image_1024.jpg", "rb") as handle:
    jpeg_bytes = handle.read()

# BYTES tensor of shape [1, 1]
jpeg_input = grpcclient.InferInput("jpeg_image", [1, 1], "BYTES")
jpeg_input.set_data_from_numpy(np.array([[jpeg_bytes]], dtype=object))

result = client.infer("sam2.1_large_encoder", inputs=[jpeg_input])
image_embed = result.as_numpy("image_embed")          # (1, 256, 64, 64) float32

# --- raw tensor path -------------------------------------------------------
# Client-side normalization, matching the table above.
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

rgb = ...                                              # (1024, 1024, 3) uint8 RGB
pixels = (rgb.astype(np.float32) / 255.0 - mean) / std
tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])   # (1, 3, 1024, 1024)

raw_input = grpcclient.InferInput("image", list(tensor.shape), "FP32")
raw_input.set_data_from_numpy(tensor)
result = client.infer("sam2.1_large_encoder", inputs=[raw_input])
```

For `sam2.1_large_encoder_fp16` the identical call returns `float16` arrays —
only the dtype changes.

## Notes

- **Raise the gRPC max message size.** The FP32 encoder response is ~16.8 MB,
  well over the 4 MiB default. The `_fp16` endpoint halves it but still exceeds
  the default.
- **Cancellation works.** Cancelling the gRPC call drops the request from the
  queue without running the encoder, which is the point of the front-end model.
  This is what the removed `sam2_encoder_jpeg` ensemble could never do — Triton
  does not propagate cancellation to already-dispatched ensemble children.
- **Request priority** is available via the batching scheduler's two levels
  (1 jumps the queue, 2 is the default).
