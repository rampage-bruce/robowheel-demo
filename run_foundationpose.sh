#!/bin/bash
# Run FoundationPose in Docker
# Usage: sudo bash run_foundationpose.sh

WORKSPACE=/mnt/users/yjy/robowheel-demo

sudo docker run --gpus '"device=0"' --rm \
  -v ${WORKSPACE}/FoundationPose:/workspace/FoundationPose \
  -w /workspace/FoundationPose \
  foundationpose_cuda124 \
  bash -c "
    echo '=== Building C++ extensions ==='
    cd mycpp && mkdir -p build && cd build && \
    cmake -DCMAKE_PREFIX_PATH=\$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())') .. && \
    make -j8 && cd ../..
    echo 'Extensions built!'

    echo '=== Running FoundationPose ==='
    python3 run_demo.py \
      --mesh_file demo_data/pick_bottle/mesh.obj \
      --test_scene_dir demo_data/pick_bottle \
      --est_refine_iter 5 \
      --track_refine_iter 2
  "
