#!/bin/bash
# Run SPIDER MJWP inside Docker with CUDA 12.4
# Mounts host conda env + code, avoids rebuilding from scratch

SPIDER_DIR=/mnt/users/yjy/robowheel-demo/spider
CONDA_ENV=/mnt/users/yjy/miniconda3/envs/spider
OUTPUT_DIR=/mnt/users/yjy/robowheel-demo/output

sg docker -c "docker run --rm --gpus all \
  -v ${SPIDER_DIR}:/workspace/spider \
  -v ${CONDA_ENV}:/opt/conda_env \
  -v ${OUTPUT_DIR}:/workspace/output \
  -v /mnt/users/yjy/.cache:/root/.cache \
  -w /workspace/spider \
  -e MUJOCO_GL=egl \
  -e CUDA_VISIBLE_DEVICES=0 \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash -c '
    # Use host conda env python + packages
    export PATH=/opt/conda_env/bin:\$PATH
    export LD_LIBRARY_PATH=/opt/conda_env/lib:\$LD_LIBRARY_PATH

    # Clear Warp cache (will recompile with CUDA 12.4)
    rm -rf /root/.cache/warp/

    echo \"Python: \$(python --version)\"
    echo \"CUDA: \$(nvcc --version | tail -1)\"

    python -u examples/run_mjwp.py \
      dataset_dir=example_datasets \
      dataset_name=hamer_demo \
      robot_type=allegro \
      embodiment_type=right \
      task=pick_bottle \
      data_id=0 \
      show_viewer=false \
      save_video=true \
      device=cuda:0 \
      num_samples=256
  '"
