# Triton server image extended with Pillow so the sam2_preprocess Python backend
# model can decode JPEGs. numpy already ships in the -py3 image; Pillow does not.
FROM nvcr.io/nvidia/tritonserver:25.01-py3

RUN pip install --no-cache-dir "pillow>=10.0.0"
