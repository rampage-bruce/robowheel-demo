#!/bin/bash
# Build custom FoundationPose Docker image with CUDA 12.4
# Usage: sudo bash build_foundationpose.sh

echo "Building FoundationPose Docker image (this takes ~15 min)..."
sudo docker build \
  --network host \
  -t foundationpose_cuda124 \
  -f /mnt/users/yjy/robowheel-demo/FoundationPose/docker/Dockerfile.cuda124 \
  /mnt/users/yjy/robowheel-demo/FoundationPose/docker/

echo "Done! Run with:"
echo "  sudo bash /mnt/users/yjy/robowheel-demo/run_foundationpose.sh"
