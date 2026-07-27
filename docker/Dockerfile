# Python runtime environment only — project code and data are bind-mounted
# at runtime (see slurm/train.sh). Mirrors the versions pinned in
# requirements-lock.txt, generated from the working .venv (Python 3.10.12,
# torch 2.13.0+cu130).
#
# Uses a CUDA devel base (not just python:slim) because deepspeed's
# op_builder calls installed_cuda_version() at `import deepspeed` time,
# which requires nvcc/CUDA_HOME to exist even when no custom op is built.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04's default python3 is already 3.10.12, matching the .venv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r /tmp/requirements-lock.txt && \
    rm /tmp/requirements-lock.txt

WORKDIR /workspace
