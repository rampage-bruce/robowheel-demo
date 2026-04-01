# RoboWheel Demo

RoboWheel 简化版 Pipeline 演示：从视频中的人手动作，经 3D 重建 + 灵巧手重定向，到 MuJoCo 物理仿真。

```
互联网视频 → HaMeR 手部 3D 重建 → MANO 参数 → 灵巧手重定向 → MuJoCo 仿真 → 视频
```

## Pipeline 总览

| 步骤 | 脚本 | 说明 |
|------|------|------|
| Step 1 | `hamer/` (子模块) | HaMeR 手部 3D 重建，输出 MANO 参数 |
| Step 2 | `hoi_sim_demo.py` | HOI 3D 场景重建（手 + 物体 + 桌面） |
| Step 3 | `step_dexterous_sim.py` | MANO → Shadow Hand 重定向 + MuJoCo 仿真 |
| Step 3 变体 | `step_allegro.py` | MANO → Allegro Hand |
| Step 3 变体 | `step_multi_hand.py` | 多灵巧手对比（Shadow / Allegro / LEAP） |
| 精化 | `step_approach_refine.py` | 接近轨迹优化 |
| 精化 | `step_sdf_refine.py` | SDF 碰撞避免 |

## 快速开始（Quick Start）

### 1. 克隆仓库

```bash
git clone --recurse-submodules git@github.com:12vv/robowheel-demo.git
cd robowheel-demo
```

如果已经 clone 过但没有拉子模块：

```bash
git submodule update --init --recursive
```

### 2. 安装依赖

```bash
# 推荐 Python 3.10+，需要 GPU（CUDA 11.7+）
pip install numpy scipy mujoco opencv-python
```

渲染需要 EGL 支持（headless GPU 渲染）：

```bash
# Ubuntu/Debian
sudo apt-get install libegl1-mesa libegl-dev libgl1-mesa-glx
```

### 3. 运行测试（无需 GPU 模型推理）

仓库自带 `test_data/mano_results.json`（预计算的 MANO 手部参数），可以直接跑 MuJoCo 仿真：

```bash
# 确保 headless 渲染
export MUJOCO_GL=egl

# 运行 Shadow Hand 仿真 demo
python step_dexterous_sim.py
```

输出在 `output/dexterous_sim/`：
- `shadow_hand_grasp.mp4` — Shadow Hand 抓瓶子仿真
- `shadow_retarget.json` — 151 帧 × 20 actuator 控制数据

### 4. 运行其他 demo

```bash
# Allegro Hand 仿真
python step_allegro.py

# 多灵巧手对比 (Shadow / Allegro / LEAP)
python step_multi_hand.py

# 完整 v2/v3 变体（含多视角、三栏对比）
python step_dexterous_v2.py
python step_dexterous_v3.py
```

## 完整 Pipeline（从视频开始）

如需从头跑完整流程（包含 HaMeR 模型推理），需要额外安装 HaMeR 环境：

```bash
# 1. 安装 HaMeR
pip install -e hamer/.[all]
pip install -v -e hamer/third-party/ViTPose

# 2. 下载模型权重
cd hamer && bash fetch_demo_data.sh && cd ..

# 3. 运行手部重建（需要 GPU ~2.6GB）
python hamer/demo.py --vid_file your_video.mp4 --out_dir output/your_video

# 4. 运行 HOI 重建（需要 pyrender, trimesh, torch）
pip install pyrender trimesh torch
python hoi_sim_demo.py
```

## Docker（可选）

```bash
docker build -f docker/Dockerfile.spider -t robowheel .
```

## 项目结构

```
robowheel-demo/
├── step_dexterous_sim.py     # 核心：MANO → Shadow Hand → MuJoCo
├── step_allegro.py           # Allegro Hand 变体
├── step_multi_hand.py        # 多灵巧手对比
├── step_dexterous_v2.py      # 多视角渲染变体
├── step_dexterous_v3.py      # 精细化变体
├── step_approach_refine.py   # 接近轨迹优化
├── step_sdf_refine.py        # SDF 碰撞优化
├── hoi_sim_demo.py           # HOI 3D 场景重建
├── convert_mano_to_spider.py # MANO → SPIDER 格式转换
├── step1_mano_to_grasp.py    # MANO → 6DoF 抓取位姿
├── step2_curobo_plan.py      # CuRobo 运动规划
├── step3_mujoco_sim.py       # Franka 夹爪仿真（对比）
├── visualize_mujoco.py       # MuJoCo 可视化工具
├── viz_3d.py                 # 3D 手部渲染
├── test_data/                # 预计算测试数据
│   └── mano_results.json     # HaMeR 输出的 MANO 参数
├── docker/                   # Docker 配置
├── PIPELINE_PLAN.md          # 详细执行记录
├── hamer/                    # [submodule] HaMeR 手部重建
├── spider/                   # [submodule] SPIDER 灵巧手重定向
├── Hunyuan3D-2/              # [submodule] 3D 物体生成
└── mujoco_menagerie/         # [submodule] MuJoCo 机器人模型库
```

## 关键依赖

| 依赖 | 用途 |
|------|------|
| [HaMeR](https://github.com/geopavlakos/hamer) | ViT-H 手部 3D 重建 |
| [SPIDER](https://github.com/facebookresearch/spider) | Meta FAIR 灵巧手物理重定向 |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Shadow Hand / Allegro / LEAP 等机器人模型 |
| [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | 3D 物体生成（完整 pipeline 用） |
| [MuJoCo](https://mujoco.org/) | 物理仿真引擎 |

## License

本仓库代码仅供研究和演示用途。子模块各有其 License，请参阅各子模块仓库。
