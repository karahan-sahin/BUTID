#!/usr/bin/env bash
# Build the BUTID Docker image for the Blackwell (RTX PRO 6000, sm_120)
# machine and convert it to an Apptainer .sif. Run this directly on that
# machine (or anywhere with docker + apptainer, then scp the .sif over).
set -euo pipefail

IMAGE_NAME=butid-blackwell
TAG="${1:-latest}"

cd "$(dirname "$0")"

docker build -t "${IMAGE_NAME}:${TAG}" -f Dockerfile.blackwell .

SIF_PATH="${IMAGE_NAME}.sif"
apptainer build --force "${SIF_PATH}" "docker-daemon://${IMAGE_NAME}:${TAG}"

echo
echo "Built docker/${SIF_PATH}"