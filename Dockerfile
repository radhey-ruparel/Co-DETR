# CUDA 11.3 + cuDNN8 runtime matches torch 1.11.0+cu113
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
  && rm -rf /var/lib/apt/lists/*

# make "python" point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install --upgrade pip==23.3.2 wheel==0.43.0 setuptools==69.5.1

WORKDIR /workspace

# ---- Core DL stack (Torch 1.11 + cu113 wheels) ----
RUN python -m pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0

# ---- Prereqs for native builds (COCO API) ----
RUN python -m pip install --no-cache-dir numpy==1.23.5 cython==0.29.36

# ---- COCO API: prefer prebuilt mmpycocotools; fallback to pycocotools ----
RUN python -m pip install --no-cache-dir mmpycocotools==12.0.3 || \
    python -m pip install --no-cache-dir pycocotools==2.0.6

# ---- OpenMMLab pins for MMDet v2 (NOT v3) ----
RUN python -m pip install --no-cache-dir \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html \
  mmcv-full==1.7.0 && \
  python -m pip install --no-cache-dir mmengine==0.10.4 mmdet==2.25.3

# ---- Packages you requested to pin to these exact versions ----
RUN python -m pip install --no-cache-dir \
  fairscale==0.4.6 \
  einops==0.6.1 \
  fvcore==0.1.5.post20221221 \
  tensorboard==2.11.2 \
  timm==0.6.13

# ---- Remainder of your conda env (split across a few RUNs) ----
# Group 1
RUN python -m pip install --no-cache-dir \
  absl-py==2.3.1 addict==2.4.0 cachetools==5.5.2 \
  # certifi 2025.10.5 is not on PyPI; use closest:
  certifi==2024.8.30 \
  charset-normalizer==3.4.3 contourpy==1.1.1 cycler==0.12.1 \
  filelock==3.16.1 fonttools==4.57.0 \
  # fsspec 2025.9.0 is not on PyPI; use closest:
  fsspec==2024.9.0 grpcio==1.70.0 hf-xet==1.1.10 huggingface-hub==0.35.3 \
  idna==3.10 importlib-metadata==8.5.0 importlib-resources==6.4.5 iopath==0.1.10

# Group 2
RUN python -m pip install --no-cache-dir \
  kiwisolver==1.4.7 markdown==3.7 markdown-it-py==3.0.0 markupsafe==2.1.5 \
  matplotlib==3.7.5 mdurl==0.1.2 oauthlib==3.3.1 opencv-python==4.12.0.88 \
  packaging==25.0 pillow==10.4.0 platformdirs==4.3.6 portalocker==3.0.0

# Group 3
RUN python -m pip install --no-cache-dir \
  protobuf==3.20.3 pyasn1==0.6.1 pyasn1-modules==0.4.2 \
  pygments==2.19.2 pyparsing==3.1.4 python-dateutil==2.9.0.post0 pyyaml==6.0.3 \
  requests==2.32.4 requests-oauthlib==2.0.0 rich==14.2.0 rsa==4.9.1 \
  safetensors==0.5.3 scipy==1.10.1 six==1.17.0 tabulate==0.9.0

# Group 4
RUN python -m pip install --no-cache-dir \
  tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.1 \
  termcolor==2.4.0 terminaltables==3.1.10 tomli==2.3.0 tqdm==4.67.1 \
  typing-extensions==4.13.2 urllib3==2.2.3 werkzeug==3.0.6 \
  yacs==0.1.8 yapf==0.43.0 zipp==3.20.2 \
  google-auth==2.43.0 google-auth-oauthlib==0.4.6

# Copy only source code (no weights/data)
COPY tools/ /workspace/tools/
COPY projects/ /workspace/projects/
COPY mmdet/ /workspace/mmdet/
COPY mmcv/ /workspace/mmcv/
COPY setup.py /workspace/
# If your code doesn’t need all of these, copy fewer dirs.

# runtime env + helpful CUDA allocator setting
ENV PYTHONPATH=/workspace
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# convenience mount points
RUN mkdir -p /models /data /output

ENTRYPOINT ["/bin/bash","-lc"]

# ✅ KEEP CONTAINER ALIVE FOR RUNPOD SSH (so it doesn't exit immediately)
CMD ["sleep infinity"]
