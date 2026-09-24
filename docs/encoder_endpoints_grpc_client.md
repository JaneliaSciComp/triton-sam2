# Encoder endpoints: gRPC client guide

Every encoder endpoint takes **either** a raw FP32 tensor **or** a single JPEG.
Sending a JPEG saves ~12.6 MB of upload per 1024×1024 image.

Supersedes the old `sam_encoder_jpeg` guide. That endpoint, along with
`sam2_encoder_jpeg`, has been removed — JPEG support is now an optional input on
each encoder's own name, so there is no separate JPEG endpoint and no
`model_type` flag.

## Endpoints

| endpoint | raw input | shape | JPEG size | outputs | default dtype |
|---|---|---|---|---|---|
| `sam1_encoder` | `image` | 1×3×1024×1024 | 1024×1024 | `image_embeddings` | FP32 |
| `sam2.1_large_encoder` | `image` | 1×3×1024×1024 | 1024×1024 | `image_embed`<br>`high_res_feats_0`<br>`high_res_feats_1` | FP32 |
| `sam3_tracker_encoder_fp16` | `pixel_values` | 1×3×1008×1008 | 1008×1008 | `image_embeddings.0`<br>`image_embeddings.1`<br>`image_embeddings.2` | FP32 |

One endpoint per model, and **any of them will return FP16 instead if you ask**
— see [Half-precision responses](#half-precision-responses). There is no
separate `_fp16` endpoint; the earlier `sam2.1_large_encoder_fp16` was retired
in favour of the parameter.

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

## Half-precision responses

Send the request **parameter** `request_fp16: true` (a parameter, not an input
tensor) and the embeddings come back as FLOAT16, halving the response — for SAM2,
16.8 MB → 8.4 MB.

```python
result = client.infer("sam2.1_large_encoder", inputs=[raw_input],
                      parameters={"request_fp16": True})
result.as_numpy("image_embed").dtype     # float16
```

Omit it and you get FP32, **byte-for-byte what this endpoint returned before the
option existed**. That is the point of making it a parameter rather than a
second endpoint: a client that predates it doesn't send it and is unaffected.
`false`, and the string `"true"`, are both handled.

The cast is exact, not approximate. It reproduces the retired
`sam2.1_large_encoder_fp16` model bit-for-bit — that export was the same fp32
network with three `Cast`→FLOAT16 nodes bolted onto its outputs, so casting in
the front end and casting inside the graph are the same operation. Verified
against embeddings captured from the old model before it was deleted.

The input stays FP32 either way; this only affects the response.

> **The declared output dtype understates this.** `/v2/models/<name>/config`
> reports FP32 because that is the default, but the same output name carries
> FLOAT16 when you ask for it — Triton does not enforce a Python model's
> declared output dtype. If you size buffers from the model metadata rather than
> from the response, account for that.

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
| `sam2.1_large_encoder` | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
| `sam3_tracker_encoder_fp16` | 0.5, 0.5, 0.5 | 0.5, 0.5, 0.5 |

SAM1/SAM2 use the torchvision ImageNet constants — identical to SAM's canonical
`pixel_mean`/`pixel_std` of `[123.675, 116.28, 103.53]` / `[58.395, 57.12,
57.375]` divided by 255. SAM3 is not ImageNet; it just maps to [-1, 1].

**Do this yourself on the raw `image` path.** That input is passed through
untouched, so a client sending un-normalized values gets un-normalized encoder
input. The two paths only agree if the client applies the same constants the
table above lists.

Two caveats worth knowing:

- **The JPEG path does not reproduce the raw path's embeddings**, and this is
  worth understanding before switching a client over. The server's decode is
  *provably* the same arithmetic — sending one JPEG as `jpeg_image` versus
  decoding it locally and sending it raw returns **bit-identical** embeddings
  (the acceptance test gates on this). But JPEG compression itself changes
  pixels, and these encoders amplify that enormously. Measured on dev:

  | endpoint | raw vs JPEG (quality 100) |
  |---|---|
  | `sam1_encoder` | cosine 0.9994 |
  | `sam2.1_large_encoder` | cosine 0.9945 – 0.9977 |
  | `sam3_tracker_encoder_fp16` | cosine 0.9657 – 0.9777 |

  The amplification is intrinsic, not a JPEG artifact: the encoders are
  deterministic (the same tensor twice returns bit-identical output), but a
  plain gaussian perturbation at input cosine 0.99997 already drops
  `image_embed` cosine to 0.949. Raising JPEG quality does **not** reliably
  help — the relationship is not monotonic, because the *direction* of the
  perturbation matters more than its size.

  Practical upshot: cosine on these embeddings is a hair-trigger metric and a
  poor proxy for "do I get the same mask". If you adopt the JPEG path, judge it
  on mask quality, not on embedding similarity — and don't spend bandwidth on
  quality 100 expecting fidelity, because quality 90 (~10x smaller) is often no
  worse.
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

Add `parameters={"request_fp16": True}` to either call to get `float16` arrays
back — only the dtype changes.

## Notes

- **Raise the gRPC max message size.** The FP32 encoder response is ~16.8 MB,
  well over the 4 MiB default. `request_fp16` halves it but still exceeds the
  default.
- **Cancellation works.** Cancelling the gRPC call drops the request from the
  queue without running the encoder, which is the point of the front-end model.
  This is what the removed `sam2_encoder_jpeg` ensemble could never do — Triton
  does not propagate cancellation to already-dispatched ensemble children.
- **Request priority** is available via the batching scheduler's two levels
  (1 jumps the queue, 2 is the default).
