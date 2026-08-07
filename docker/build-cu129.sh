#!/usr/bin/env bash
# Build the BUTID image for Blackwell GPUs on the HPC cluster (CUDA 12.9) and
# convert it to an Apptainer .sif. Run this on a machine with both docker and
# apptainer, then scp the resulting .sif to the HPC.
set -euo pipefail

IMAGE_NAME=butid-cu129
TAG="${1:-latest}"

cd "$(dirname "$0")"

docker build -t "${IMAGE_NAME}:${TAG}" -f Dockerfile.cu129 .

SIF_PATH="${IMAGE_NAME}.sif"
apptainer build --force "${SIF_PATH}" "docker-daemon://${IMAGE_NAME}:${TAG}"

echo
echo "Built docker/${SIF_PATH}"
echo "Copy it to the HPC, e.g.:"
echo "  scp docker/${SIF_PATH} <user>@<hpc-host>:/path/to/BUTID/docker/"
echo
echo "Then confirm it matches the target GPU before submitting a long job:"
echo "  apptainer exec --nv docker/${SIF_PATH} python3 -c \\"
echo "    'import torch; print(torch.version.cuda, torch.cuda.get_arch_list())'"
