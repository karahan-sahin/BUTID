#!/usr/bin/env bash
# Build the BUTID Docker image and convert it to an Apptainer .sif locally.
# Run this on a machine with both docker and apptainer (e.g. your workstation),
# then scp the resulting .sif to the HPC.
set -euo pipefail

IMAGE_NAME=butid
TAG="${1:-latest}"

cd "$(dirname "$0")"

docker build -t "${IMAGE_NAME}:${TAG}" -f Dockerfile .

SIF_PATH="${IMAGE_NAME}.sif"
apptainer build --force "${SIF_PATH}" "docker-daemon://${IMAGE_NAME}:${TAG}"

echo
echo "Built docker/${SIF_PATH}"
echo "Copy it to the HPC, e.g.:"
echo "  scp docker/${SIF_PATH} <user>@<hpc-host>:/path/to/BUTID/docker/"
