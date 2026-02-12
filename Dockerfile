FROM nvcr.io/nvidia/pytorch:26.01-py3
WORKDIR /workspace/exaggeration-detection
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
