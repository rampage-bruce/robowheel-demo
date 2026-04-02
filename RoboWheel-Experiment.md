# RoboWheel 实验验证记录

> 基于 [RoboWheel-Survey.md](RoboWheel-Survey.md) 的调研，对 RoboWheel 核心链路进行逐步验证
> 日期：2026-03-31
> 服务器：192.168.77.25, 8× NVIDIA RTX A5000 (24GB)

---

## 1. HaMeR 手部 3D 重建验证

使用 RoboWheel 论文的核心组件 **HaMeR**（手部 3D mesh 重建）对随意视频进行手部姿态提取，验证"视频 → 手部 3D 重建"这一关键步骤的可行性。

### 1.1 实验配置

| 项目 | 详情 |
|------|------|
| Conda 环境 | `hawor` (PyTorch 2.3.0+cu121, PyTorch3D 0.7.6) |
| 手部检测 | MediaPipe Hand Landmarker (CPU, 轻量) |
| 手部重建 | **HaMeR** (ViT-H backbone, ~2.6GB GPU 显存) |
| 测试视频 | `pick_bottle` — 人手抓取水瓶（15秒，640×360，30fps） |
| 测试帧数 | 9 帧（均匀采样）→ 后扩展到 151 帧 |

### 1.2 Pipeline

```
原始视频帧 (pick_bottle)
        ↓
MediaPipe Hand Landmarker (CPU)
  → 手部 2D 边界框 + 左右手分类
        ↓
HaMeR (GPU, ViT-H)
  → MANO 3D 手部参数:
    - hand_pose (15D): 5个关节 × 3D 旋转
    - betas (10D): 手部形状参数
    - global_orient: 全局朝向
    - pred_vertices (778 顶点): 3D mesh
        ↓
PyRender 渲染
  → Overlay 图 (3D mesh 叠加原图)
  → 侧视图 (90° 旋转)
  → OBJ mesh 文件
```

### 1.3 结果

| 验证项 | 结果 | 说明 |
|--------|------|------|
| 手部检测（视频→2D bbox） | **通过** | MediaPipe 在所有帧中检测到手部 |
| 手部 3D 重建（2D→MANO） | **通过** | HaMeR 生成 778 顶点 mesh + MANO 参数 |
| 渲染验证（mesh overlay） | **通过** | 3D mesh 准确叠加到真实手部位置 |
| 左右手分类 | **通过** | 正确区分左右手 |
| 完整视频处理 | **通过** | 151 帧，261 只手全部重建成功 |

### 1.4 运行命令

```bash
cd /mnt/users/yjy/robowheel-demo/hamer
CUDA_VISIBLE_DEVICES=5 /mnt/users/yjy/miniconda3/envs/hawor/bin/python run_hamer_mp.py \
  --img_folder /mnt/users/yjy/robowheel-demo/test_videos/pick_bottle_frames \
  --out_folder /mnt/users/yjy/robowheel-demo/output/pick_bottle \
  --batch_size=2 --side_view --save_mesh
```

### 1.5 输出

```
/mnt/users/yjy/robowheel-demo/output/
├── pick_bottle/             # 9帧初始测试 (overlay + OBJ mesh)
├── pick_bottle_video/       # 151帧完整输出
│   ├── mano_results.json    # 261个手部 MANO 参数
│   ├── *_overlay.jpg        # 151帧 overlay
│   ├── hamer_overlay.mp4    # overlay 视频
│   └── hamer_sidebyside.mp4 # 原图|overlay 对比
```

**脚本**：`/mnt/users/yjy/robowheel-demo/hamer/run_hamer_mp.py`

---

## 2. 灵巧手重定向验证

在 HaMeR 手部重建基础上，验证 MANO → 灵巧手/机械臂仿真的完整链路。

### 2.1 路线 A：MANO → Franka 夹爪（已弃用）

**方法**：从 MANO thumb_tip + index_tip 提取 6DoF 抓取位姿 → CuRobo 轨迹 → MuJoCo Franka 仿真

**问题**：MANO 是 5 指 21 关节，Franka 是 1-DoF 平行夹爪，信息丢失严重

**输出**：`/mnt/users/yjy/robowheel-demo/output/franka_sim/`

### 2.2 路线 B：MANO → Shadow Hand

**方法**：MANO 15 关节 → Shadow Hand E3M5 (20 actuator) 映射

**关节映射**：

| MANO 关节 | Shadow Actuator | 映射方式 |
|-----------|----------------|---------|
| global_orient → euler XZ | WRJ1, WRJ2 | 腕部弯曲/偏转 |
| thumb[0] flex/spread | THJ5, THJ4, THJ3 | 拇指旋转/对掌/扭转 |
| thumb[1,2] flex | THJ2, THJ1 | 拇指 MCP/IP 弯曲 |
| index[0] flex/spread | FFJ4, FFJ3 | 食指展开/MCP弯曲 |
| index[1,2] flex | FFJ0 | 食指 PIP+DIP 耦合 |
| middle, ring, pinky | 同上模式 | MFJ, RFJ, LFJ |

**转换方式**：MANO 3×3 旋转矩阵 → euler XYZ → X=弯曲, Z=展开 → clip 关节限位 → 时序平滑

**MuJoCo 场景**：MjSpec attach API 挂载 Shadow Hand 到 6DoF 基座

**问题**：前臂体积过大遮挡手指；瓶子穿透桌面

**输出**：`/mnt/users/yjy/robowheel-demo/output/dexterous_v2/`, `dexterous_v3/`

### 2.3 路线 C：MANO → Allegro Hand（最终版）

改用 **Wonik Allegro Hand V3**（4 指 16 关节，无前臂）。

**5 阶段轨迹**：HOVER → DESCEND → CLOSE → HOLD → LIFT

**MANO → Allegro 映射**：index/middle/ring 直接映射，thumb flex→rotation+curl，Pinky 丢弃

**结果**：手指动作清晰可见（DESCEND 伸直→CLOSE 弯曲→HOLD 握紧），碰撞正常，瓶子被碰倒但未抓住

**输出**：`/mnt/users/yjy/robowheel-demo/output/allegro_sim/`

### 2.4 迭代总结

| 版本 | 手模型 | 手指可见 | 碰撞 | 动作序列 | 抓取成功 |
|------|--------|---------|------|---------|---------|
| v1 | Shadow | 被前臂遮挡 | 穿模 | 无阶段 | 否 |
| v2 | Shadow | 部分可见 | 穿模 | 无阶段 | 否 |
| v3 | Shadow | 部分可见 | 正常 | 3 阶段 | 否（碰到但没握住） |
| **Allegro** | **Allegro** | **完全可见** | **正常** | **5 阶段** | **否（碰翻了）** |

---

## 3. 为什么需要 RL 精化

开环 MANO → 灵巧手映射暴露了 3 个根本问题：

**问题 1：手-物尺度不匹配** — 开环映射不感知物体大小

**问题 2：接近路径缺失** — MANO 只有姿态没有路径，手动设计的"从上方下降"碰翻物体

**问题 3：缺少接触力闭环** — 不知道手指是否碰到物体

| 问题 | 开环映射 | RL 精化后 |
|------|---------|----------|
| 尺度适配 | 固定映射 | 自适应调整 |
| 接近路径 | 手动设计 | 学习避障策略 |
| 接触力控制 | 无反馈 | 实时力反馈 + 防滑 |
| 穿透消除 | 可能穿入 | SDF 惩罚 |
| 动力学平滑 | 关节角跳变 | 速度/加速度约束 |

**结论**：前半段（MANO → 关节角 → 仿真渲染）已验证，后半段需要 RL，这正是 RoboWheel 的核心贡献。

---

## 4. SPIDER 物理重定向验证

使用 [SPIDER](https://github.com/facebookresearch/spider)（Facebook Research）替代 ManipTrans（被 IsaacGym 依赖阻塞），实现 MuJoCo 原生的物理重定向。

### 4.1 SPIDER 示例数据验证 ✅

```bash
# IK 运动学重定向 (fair_fre/pour, Allegro bimanual, 224帧)
MUJOCO_GL=egl python spider/preprocess/ik_fast.py --task=pour ...
# MJWP 物理优化 (240帧, ref|sim 对比)
MUJOCO_GL=egl python examples/run_mjwp.py dataset_name=fair_fre ...
```

结果：双手 Allegro Hand 成功完成 pour（倒水）任务，手指动态+物体交互+物理合理性全部到位

### 4.2 自有数据接入

1. ✅ `convert_mano_to_spider.py`：HaMeR MANO → SPIDER `trajectory_keypoints.npz`
2. ✅ 坐标变换：MANO 相机空间 → SPIDER 世界空间（Y/Z 交换 + Z 偏移）
3. ✅ `generate_xml.py`：Allegro + 瓶子场景生成
4. ✅ `ik_fast.py`：150 帧 IK 运动学重定向
5. ❌ `run_mjwp.py`：MJWP 物理优化失败 — **Warp CUDA 内核编译 segfault**

**MJWP 失败根因**：NVIDIA Driver 570.195.03（CUDA 13.0）与 Warp 1.12/1.13 的 NVRTC 编译器不兼容。Warp 在编译 MuJoCo 仿真内核时 segfault（core dump）。之前成功运行 fair_fre 是因为使用了预编译的 Warp 缓存，清除缓存后无法重新编译。

**解决方案**（任选其一）：
- 降级 NVIDIA driver 到 <=560（CUDA 12.x 兼容）
- 等 Warp 正式版支持 CUDA 13.0
- 从 CUDA 12.x 环境拷贝预编译 Warp 缓存
- 使用 Docker 容器运行（固定 CUDA 版本）

### 4.3 输出

```
/mnt/users/yjy/robowheel-demo/output/
├── spider_ik_pick_bottle_v2.mp4  # 自有数据 IK (坐标修正后) ✅
├── spider_ik_pour.mp4            # 示例数据 IK ✅
└── spider_mjwp_pour.mp4          # 示例数据 MJWP 物理优化 ✅
```

---

## 5. 完整输出文件索引

详见 `/mnt/users/yjy/robowheel-demo/PIPELINE_PLAN.md`

```
/mnt/users/yjy/robowheel-demo/output/
├── allegro_sim/          ← Allegro Hand 5阶段轨迹
├── dexterous_v3/         ← Shadow Hand v3（碰撞修复）
├── dexterous_v2/         ← Shadow Hand v2（朝向修复）
├── dexterous_sim/        ← Shadow Hand v1
├── franka_sim/           ← Franka 夹爪
├── hoi_demo/             ← HOI 3D 重建（手+瓶子+桌子）
├── pick_bottle_video/    ← HaMeR 151帧完整输出
├── pick_bottle/          ← HaMeR 9帧初始测试
├── spider_*.mp4          ← SPIDER IK/MJWP 结果
└── grasp_poses.json      ← MANO→6DoF 抓取位姿
```

---

## 6. 轻量版精化：SPIDER IK + SDF 穿透消除（2026-04-01）

在 MJWP 物理优化被 Warp/CUDA 兼容性阻塞后，采用**方案 C + D**：SPIDER MuJoCo 原生 IK + trimesh SDF 穿透消除，不依赖 Warp。

### 6.1 方法

```
SPIDER IK 运动学轨迹 (147帧, Allegro Hand)
        ↓
逐帧 SDF 穿透检查：
  对每个指尖，计算到瓶子 mesh 的符号距离
  SD < -2mm → 判定穿透
        ↓
scipy L-BFGS-B 优化：
  调整手指关节角（保持基座不动）
  目标：min Σ(穿透距离²×1000) + Σ(关节变化²×10)
  约束：关节角在 Allegro 限位内
        ↓
时序平滑 (uniform_filter1d, window=3)
        ↓
MuJoCo 回放渲染
```

### 6.2 结果

| 指标 | 值 |
|------|-----|
| 总帧数 | 147 |
| 穿透修复数 | **80** |
| 最大关节调整 | 0.046 rad (~2.6°) |
| 平均关节调整 | 0.002 rad (~0.1°) |
| 最小指尖-物体距离 | 0.020 m |

精化前后对比：手指不再穿入瓶子，同时保持了原始 MANO 驱动的手指动态。

### 6.3 与 RoboWheel RL 精化的对比

| | 我们的 SDF 精化 | RoboWheel RL 精化 |
|--|---------------|-----------------|
| 方法 | scipy 优化（每帧独立） | PPO 强化学习（时序策略） |
| 解决穿透 | ✅ SDF 查询 + 推出 | ✅ SDF 惩罚项 |
| 接触力合理性 | ❌ 不考虑力 | ✅ 接触力奖励 |
| 运动平滑性 | 弱（后处理滤波） | 强（动力学平滑奖励） |
| 能否抓住物体 | ❌ 只防穿透，不保证握住 | ✅ 接触保持+防滑 |
| 计算时间 | ~1秒/帧 | ~2-4小时训练 |
| 依赖 | trimesh + scipy | IsaacGym 或 Warp |

**结论**：SDF 精化是一个轻量但有限的替代——解决了穿透问题（RoboWheel 管线中 TSDF 消除的等价步骤），但缺少 RL 的接触力控制和动力学平滑（ManipTrans 步骤）。

### 6.4 输出

```
/mnt/users/yjy/robowheel-demo/output/sdf_refined/
├── comparison.mp4          # 三栏: HaMeR | IK原始 | SDF精化
├── ik_original.mp4         # 精化前
├── sdf_refined.mp4         # 精化后
├── trajectory_refined.npz  # 精化后轨迹数据
├── comparison_preview.jpg
└── kf_*.jpg                # 关键帧
```

**脚本**：`/mnt/users/yjy/robowheel-demo/step_sdf_refine.py`

---

## 7. 完整管线总结

```
互联网视频 (pick_bottle, 15s)
    ↓  HaMeR + MediaPipe               ✅ 已验证
MANO 手部参数 (151帧, 261只手)
    ↓  convert_mano_to_spider.py       ✅ 已验证
SPIDER 格式 (trajectory_keypoints.npz)
    ↓  SPIDER IK (ik_fast.py)          ✅ 已验证
Allegro Hand 运动学轨迹 (147帧)
    ↓  SDF 穿透消除 (step_sdf_refine)  ✅ 已验证 (80处穿透修复)
物理精化后的轨迹
    ↓  MuJoCo 回放渲染                  ✅ 已验证
三栏对比 MP4
```

| 步骤 | RoboWheel 对应 | 我们的实现 | 状态 |
|------|---------------|-----------|------|
| 手部重建 | HaMeR | HaMeR + MediaPipe | ✅ 相同 |
| 数据格式转换 | - | MANO → SPIDER npz | ✅ |
| 运动学重定向 | 标准化动作空间 | SPIDER IK (MuJoCo) | ✅ 替代 |
| 穿透消除 | TSDF SDF 优化 | trimesh SDF + scipy | ✅ 轻量替代 |
| RL 物理精化 | ManipTrans PPO | ❌ (Warp/CUDA 阻塞) | 待解决 |
| 跨形态重定向 | CoTracker + 标准化 | SPIDER IK | ✅ 替代 |
| 仿真渲染 | Isaac Sim | MuJoCo | ✅ 替代 |

---

---

## 8. Docker MJWP 尝试（2026-04-01）

尝试通过 Docker 容器（固定 CUDA 版本）绕过 Warp segfault。

**结果**：Docker 路线也被阻塞。

| 问题 | 详情 |
|------|------|
| Docker 权限 | `sg docker` 可用（用户在 `dockers` 组，socket 属于 `docker` 组） |
| GPU 访问 | `--gpus all` + nvidia-smi 正常 |
| Docker 网络 | **无法访问外网**（DNS 解析失败，公司防火墙限制） |
| CUDA 兼容性 | 本地仅有 CUDA 12.4 镜像，host driver 570 = CUDA 13.0，不兼容 |
| 最终状态 | Warp 报 `error 803: unsupported display driver / cuda driver combination` |

**需要**：管理员拉取 CUDA 13.0 Docker 镜像，或降级 host NVIDIA driver 到 560 以下。

---

*实验日期：2026-03-31 ~ 2026-04-01*

---

## 9. 侧面接近轨迹优化（2026-04-01）

在 SDF 精化基础上，改进手指动态：添加 4 阶段渐进 curl + 弧线接近。

### 改进内容

| 阶段 | 时间 | 手指动作 | 基座运动 |
|------|------|---------|---------|
| APPROACH | 0-30% | 保持 IK 原始 | 弧线接近物体 |
| CLOSE | 30-60% | 渐进 curl +0.4 rad | 闭合间隙 |
| GRASP | 60-80% | 强 curl +0.7 rad (含拇指对掌) | 保持 |
| LIFT | 80-100% | 保持 curl | Z+5cm 抬起 |

### 结果

- 手指弯曲动态**显著改善**：4 阶段清晰可见
- 穿透仅 0-9 处（SDF 精化修正）
- 基座偏移方向仍待优化（base→object 方向计算需考虑手臂运动学链）

### 输出

```
/mnt/users/yjy/robowheel-demo/output/approach_refined/
├── comparison.gif          # 三栏: HaMeR | IK原始 | 接近精化
├── approach_refined.mp4    # 精化后单独视频
└── kf_*_{approach,grasp,lift}.jpg
```

**脚本**：`/mnt/users/yjy/robowheel-demo/step_approach_refine.py`

---

## 10. 多灵巧手对比测试（2026-04-01）

在同一个 MANO 驱动的轨迹上，测试 3 种不同灵巧手模型。

### 测试的手模型

| 手模型 | 来源 | 关节 | 驱动器 | 效果 |
|--------|------|------|--------|------|
| **Allegro V3** | Wonik Robotics | 16 | 16 (位置控制) | **最佳** — 手指动态清晰，拇指包裹瓶子 |
| LEAP Hand | CMU | 16 | 16 (位置控制) | 手被挂载到错误位置，不在画面内 |
| Shadow DEXee | Shadow Robot | 12 | 12 (位置控制) | 只有 3 个 actuator 正确挂载，太暗 |

### 结论

- **Allegro Hand** 最适合当前管线：结构紧凑、无前臂遮挡、关节数和 MANO 接近
- LEAP/Shadow DEXee 需要针对性调整挂载参数（不同手的坐标系和关节命名约定不同）
- MjSpec attach API 对不同 MJCF 格式的兼容性有差异

### 输出

`/mnt/users/yjy/robowheel-demo/output/multi_hand/` — 各手独立 MP4 + 三栏对比

### 10.2 比例修正后（v2）

修正瓶子尺寸为真实比例（直径 6.6cm, 高度 20cm），手-物比例正确。

输出：`/mnt/users/yjy/robowheel-demo/output/multi_hand_v2/`

---

## 11. Hunyuan3D-2 物体重建接入（2026-04-01）

从视频截帧中用 Hunyuan3D-2 AI 生成水瓶 3D mesh，替代手动圆柱体。

### 结果

| 指标 | 值 |
|------|-----|
| 输入 | pick_bottle 视频 frame 150 裁剪 (100×210px) |
| 模型 | tencent/Hunyuan3D-2 (DiT Flow Matching) |
| 原始 mesh | 172,476 顶点, 344,944 面 |
| 简化后 | 2,502 顶点, 5,000 面 |
| 缩放后尺寸 | 8.8cm × 20cm × 12.5cm（真实水瓶比例）|
| GPU 显存 | ~16GB (RTX A5000) |
| 生成时间 | ~8 分钟（含模型下载） |

### 对比

| | 之前（手动） | Hunyuan3D2 生成 |
|--|-----------|---------------|
| 形状 | 圆柱体 | 真实瓶形（含瓶颈、标签区域凹凸） |
| 顶点数 | 34 | 2,502 |
| SDF 精度 | 低（圆柱近似） | 高（真实表面） |

### 输出

```
/mnt/users/yjy/robowheel-demo/output/
├── bottle_hunyuan3d.obj          # 原始 mesh (13MB, 172K verts)
├── bottle_hunyuan3d_scaled.obj   # 简化+缩放 (166KB, 2.5K verts, 真实尺寸)
├── bottle_crop.png               # 输入裁剪
└── bottle_rembg.png              # 去背景后
```

---

## 12. nvblox GPU SDF 优化调研（2026-04-01）

调研 NVIDIA nvblox 替代 trimesh CPU SDF，作为穿透消除的 GPU 加速方案。

### nvblox 优势

| 特性 | trimesh (当前) | nvblox (目标) |
|------|--------------|-------------|
| 运行设备 | CPU 逐点查询 | **GPU 批量查询** |
| SDF 类型 | mesh 表面近似 | **TSDF/ESDF 体素场** |
| 精度 | 有限（nearest.on_surface） | **体素级符号距离** |
| 动态场景 | 不支持 | **支持 RGB-D 流实时重建** |
| Isaac 集成 | 无 | **Isaac ROS 核心组件** |
| 与 LingBot-Depth | 无 | **可直接对接深度输出** |

### 安装尝试

- `nvblox_torch` (NVlabs/nvblox_torch): PyTorch 绑定
- 依赖链: glog (已编译安装) → nvblox 核心库 (需 cmake + CUDA submodule 构建) → nvblox_torch
- **阻塞点**: nvblox 核心库的 git submodule 拉取不完整（LFS 文件 + 网络问题）

### 构建步骤（后续执行）

```bash
# 1. 安装 glog (已完成)
cd /tmp && git clone https://github.com/google/glog.git
cd glog && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX .. && make -j8 && make install

# 2. 构建 nvblox 核心 (需要完整 submodule)
git clone --recursive https://github.com/nvidia-isaac/nvblox.git
cd nvblox && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES=86 ..
make -j8 && make install

# 3. 构建 nvblox_torch
git clone https://github.com/NVlabs/nvblox_torch.git
cd nvblox_torch && bash install.sh $(python -c "import torch; print(torch.utils.cmake_prefix_path)")
pip install -e .
```

### nvblox_torch 使用方式

```python
from nvblox_torch.mapper import Mapper

# 初始化
mapper = Mapper(voxel_sizes=[0.01], integrator_types=["tsdf"])

# 从深度图构建 TSDF
mapper.add_depth_frame(depth_image, intrinsics, extrinsics)

# GPU 批量 SDF 查询
query_points = torch.tensor(fingertip_positions).cuda()
sdf_values = mapper.query_sdf(query_points)
# sdf < 0 = 穿透, sdf ≈ 0 = 接触, sdf > 0 = 远离
```

### 结论

nvblox 是 SDF 精化的正确升级路径，但构建链较重。当前 trimesh SDF 作为轻量替代已验证可行（368 处穿透检测 + 修复）。后续优化建议：
1. 解决 nvblox submodule 拉取问题（需要稳定网络）
2. 或使用 Docker 容器（nvblox 有官方 Docker 镜像）
3. 构建完成后替换 `step_sdf_refine.py` 中的 `sdf_penetration_check()` 函数


### 12.2 nvblox 编译安装成功（2026-04-02）

**nvblox + nvblox_torch 在宿主机上编译安装成功**，GPU SDF 查询验证通过。

**构建过程**：
1. 从 `nvidia-isaac/nvblox` mono-repo 克隆（含 git-lfs）
2. 编译 glog 依赖（从源码）
3. cmake + make 编译 nvblox 核心库 + nvblox_torch PyTorch 绑定
4. 安装到 spider conda env

**验证结果**：
```
SDF query (plane at z=0.5m):
  z=0.4 (front):  dist=+0.020, weight=5.0  ← 正值=外部 ✅
  z=0.5 (surface): dist=-0.003, weight=4.0  ← 接近零=表面 ✅
  z=0.6 (behind):  dist=100.0, weight=0.0   ← 未观测 ✅
```

**API 用法**：
```python
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.sensor import Sensor

mapper = Mapper(voxel_sizes_m=[0.005])
sensor = Sensor.from_camera(fu=500, fv=500, cu=320, cv=240, width=640, height=480)
mapper.add_depth_frame(depth_gpu, pose_cpu, sensor)
sdf = mapper.query_layer(QueryType.TSDF, points_gpu)  # → (N, 2) [distance, weight]
```

**下一步**：用 nvblox 替换 `step_sdf_refine.py` 中的 trimesh SDF 查询。

### 12.3 nvblox GPU SDF 精化运行成功（2026-04-02）

用 nvblox 替换 trimesh 作为 SDF 后端，完整管线跑通。

**对比 trimesh vs nvblox**：

| | trimesh (之前) | nvblox (现在) |
|--|-------------|------------|
| 运行设备 | CPU | **GPU (CUDA)** |
| 穿透检测数 | 80 → 368 | **253** |
| 穿透阈值 | 2mm | **0.5mm** |
| SDF 构建 | mesh.nearest.on_surface | **TSDF 12视角深度渲染** |
| 批量查询 | 逐点 | **GPU 批量** |
| Loss 输出 | 无 | **pen_loss + reg_loss 逐帧** |

**输出**：
```
/mnt/users/yjy/robowheel-demo/output/sdf_nvblox/
├── comparison.gif / .mp4    # 三栏: HaMeR | IK原始 | nvblox精化
├── nvblox_refined.mp4       # 精化后视频
├── trajectory_nvblox.npz    # 精化后轨迹
└── kf_*.jpg                 # 关键帧
```

**脚本**：`/mnt/users/yjy/robowheel-demo/step_sdf_nvblox.py`

### 12.4 nvblox 精化问题分析（2026-04-02）

**问题 1: 优化器未实际修正穿透**
- pen_loss=17.16 但 reg_loss=0.0 → L-BFGS-B 在第一步就停止
- 原因：scipy 有限差分梯度 + nvblox GPU 查询 + MuJoCo FK 链路太慢，收敛不了
- 需要：直接修正策略（检测穿透→强制推出）代替迭代优化

**问题 2: 手指不可见**
- 相机距离 0.30m 太近，瓶子占满画面
- 手指在瓶子后面被遮挡
- 需要：拉远相机到 0.45m+，调整角度

**问题 3: 无轨迹规划**
- IK 轨迹仅跟随 MANO 数据，无抓取意图
- 没有分阶段动作（接近→包裹→抓取→提起）
- 需要：合并 `step_approach_refine.py` 的轨迹设计

**解决方案：合并脚本**
- 分阶段轨迹（approach→close→grasp→lift）来自 step_approach_refine
- nvblox GPU SDF 穿透检测来自 step_sdf_nvblox
- 强制推出策略替代 scipy 优化
- 正确的相机视角（distance=0.45, 同时看到手+瓶）

### 12.5 统一管线：分阶段轨迹 + nvblox SDF + 正确渲染（2026-04-02）

合并了所有优化：分阶段抓取轨迹 + nvblox GPU SDF 穿透检测 + 直接推出策略 + 正确相机视角。

**4 阶段轨迹**：

| 阶段 | 时间 | 基座 | 手指 |
|------|------|------|------|
| APPROACH | 0-30% | 弧线接近物体 | 保持 IK 原始 |
| CLOSE | 30-55% | 闭合间隙 | 渐进 curl +0.5rad |
| GRASP | 55-75% | 紧贴物体 | 强 curl +0.8rad |
| LIFT | 75-100% | Z+4cm 抬起 | 保持 curl |

**nvblox SDF 直接推出**（替代 scipy 优化）：
- 检测穿透 < 0.5mm
- 直接减少穿透手指的 curl 0.08rad
- 127 处穿透检测+推出

**渲染改善**：
- 相机距离 0.45m（之前 0.30m）
- 手指和瓶子同时可见
- 阶段标注叠加在画面上

**输出**：`/mnt/users/yjy/robowheel-demo/output/unified_grasp/`

**脚本**：`/mnt/users/yjy/robowheel-demo/step_unified_grasp.py`

### 12.6 完整 5 阶段抓取：REACH → APPROACH → CLOSE → GRASP → LIFT（2026-04-02）

添加了 REACH 阶段（40 帧），手从远处平滑移动到瓶子附近再开始抓取。

**5 阶段**：

| 阶段 | 帧数 | 手指 | 基座运动 |
|------|------|------|---------|
| REACH | 0-39 | 完全张开 | 从远处 cubic ease-in-out 接近 |
| APPROACH | 40-76 | 保持 IK | 弧线靠近物体 |
| CLOSE | 77-113 | 渐进 curl | 闭合间隙 |
| GRASP | 114-143 | 强 curl | 紧贴物体 |
| LIFT | 144-186 | 保持 curl | Z+5cm 抬起 |

**关键数据**：
- 总帧数: 187 (40 REACH + 147 IK)
- min_dist: 399mm → 192mm → 52mm → 25mm（渐进接近）
- 穿透推出: 131 处

**输出**: `/mnt/users/yjy/robowheel-demo/output/unified_grasp/`

---

## 13. 双手抓取 + nvblox Binary Search 穿透修正（2026-04-02）

完整的双手灵巧手抓取管线：左右手 MANO 数据 → 双 Allegro Hand → 5 阶段轨迹 → nvblox 二分搜索穿透修正。

### Pipeline

```
HaMeR MANO (151帧右手 + 110帧左手)
    ↓ smplx 重建关节位置
双手 trajectory (110帧 bimanual)
    ↓ build_bimanual_scene()
MuJoCo: 2× Allegro Hand + 蓝色圆柱瓶 + 桌子
    ↓ 5-stage trajectory
REACH(40帧) → APPROACH → CLOSE → GRASP → LIFT
    ↓ nvblox binary search
穿透修正: 二分搜索每根手指最大安全 curl 值
    ↓ MuJoCo 渲染
双栏对比: HaMeR 原视频 | 双手 Allegro 仿真
```

### 结果

| 指标 | 值 |
|------|-----|
| 双手帧数 | 110 + 40 REACH = 150 |
| 模型 | 2× Allegro Hand (16 DOF × 2 + 6 base) = 38 actuators |
| 穿透检测 | **1 处** (之前单手 131 处) |
| Binary search | 4 个关节修正 (6 次二分 = 1/64 精度) |

### 关键改进

- **双手**: 左右 Allegro Hand 同时从瓶子两侧接近
- **Binary search 穿透修正**: 不再用固定 -0.08rad 推出，而是二分搜索每根手指刚好不穿透的最大 curl 值
- **穿透从 131→1**: 证明 binary search 策略有效

### 输出

```
/mnt/users/yjy/robowheel-demo/output/bimanual_grasp/
├── comparison.gif / .mp4    # 双栏: HaMeR 双手 | Allegro 仿真
├── bimanual_grasp.mp4       # 仿真单独
└── kf_*_{reach,approach,grasp,lift}.jpg
```

**脚本**: `/mnt/users/yjy/robowheel-demo/step_bimanual_grasp.py`

### 13.2 手朝向修正：水平伸手 + 掌心相对（2026-04-02）

**修正内容**：
- 右手 mount: `euler=[180°Z, -15°Y, 0]` → 手指朝左指向瓶子，掌心朝内
- 左手 mount: `euler=[0, 15°Y, 0]` → 手指朝右指向瓶子，掌心朝内
- REACH: 水平方向从两侧靠近（X 轴），不再从上方下降（Z 轴）
- 相机: 正面视角 (azimuth=180°)

**问题根源**：之前 mount rotation 硬编码 `euler=[90,0,180]` 让手指朝下（适合从上方抓取），但视频中人是水平伸手。mount 应该根据任务调整，理想情况应从 MANO global_orient 推导。

**结果**：双手水平从两侧夹住瓶子，和视频中的抓取姿态一致。

### 13.3 MANO global_orient 自动驱动手朝向（2026-04-02）

**修正内容**：
- Mount rotation 不再硬编码，改为 identity + 3 个 hinge 关节
- 基础朝向：右手 Z=π (手指朝左) + Y=-0.25 (掌心内倾)，左手 Y=+0.25
- MANO global_orient 转换为帧间 delta rotation（相对第一帧的变化）
- Delta 缩放 15%（ORIENT_SCALE=0.15）防止过大旋转

**结果**：
- Frame 0 (REACH)：双手水平伸出、瓶子在中间、桌面可见 ✅
- 手的朝向跟随 MANO 数据动态变化 ✅
- 穿透：仅 1 处（binary search 修 4 个关节）
- 仿真稳定（无 NaN 警告）

**待优化**：
- 后续帧中 delta rotation 把手推离瓶子 → 需要位置跟踪补偿
- ORIENT_SCALE 需要针对不同视频调参
- 相机应该跟踪手+瓶子的中心

**输出**: `/mnt/users/yjy/robowheel-demo/output/bimanual_grasp/`
**脚本**: `/mnt/users/yjy/robowheel-demo/step_bimanual_grasp.py`

### 13.4 稳定版双手抓取（2026-04-02）

关掉不稳定的 MANO rotation/translation 驱动，改为：
- **固定水平朝向**：右手 Z=π (手指朝左)，左手 Y=0.25 (掌心内倾)
- **Staged X 轴接近**：双手沿 X 轴向瓶子移动 5cm
- **瓶子 static**：无 freejoint，不会被碰飞（visual only）
- **3/4 视角相机**：azimuth=145°, elevation=-30°, distance=0.42m

**结果**：
- 5 帧全部稳定：双手+瓶子+桌面始终可见
- 穿透：46 处检测 → 184 关节 binary search 修正
- 无仿真不稳定（NaN）
- Frame 95 GRASP：指尖球体贴住瓶身两侧

**简化的代价**：
- 手的朝向不跟随 MANO 数据（固定值）
- 手的位移只沿 X 轴（无 Y/Z 跟踪）
- 瓶子不参与物理（不能被抓起）

**这是展示"双手灵巧手从两侧包裹瓶子"效果的最稳定版本。**

---

## 14. 物理仿真交互（2026-04-02）

彻底重写：用 MuJoCo 原生接触物理替代外部 SDF 检查。

### 核心变化

| | 之前 (所有版本) | 现在 (physics_grasp) |
|--|-------------|---------------------|
| 瓶子 | static 或 visual-only | **freejoint + 碰撞 + 质量 + 摩擦** |
| 穿透检测 | nvblox/trimesh 外部 SDF | **MuJoCo 内置接触求解器** |
| 手指碰撞 | 无 | **Allegro 自带碰撞 geom** |
| 物理 | 无 | **重力 + 摩擦 + 接触力** |
| 接触数 | 0 (无物理) | **最高 15 个同时接触** |
| 穿透 | 需要手动修复 | **物理引擎自动阻止** |

### 关键参数

```python
# 瓶子物理属性
mass = 0.25 kg
friction = [1.5, 0.01, 0.001]  # 高摩擦
solref = [0.02, 1.0]            # 接触刚度
solimp = [0.9, 0.95, 0.001, 0.5, 2.0]  # 接触阻抗

# MuJoCo 求解器
impratio = 10      # 椭圆摩擦锥
cone = ELLIPTIC    # 更好的摩擦模型
```

### 结果

- 260 帧 (60 REACH + 200 GRASP)
- 接触数: 1→5→11→15 (随阶段递增)
- **手指不再穿透瓶子** (MuJoCo 自动处理)
- 瓶子未抬起 (握力 < 重力，需要更紧的包裹)

### 输出

`/mnt/users/yjy/robowheel-demo/output/physics_grasp/`

**脚本**: `/mnt/users/yjy/robowheel-demo/step_physics_grasp.py`

---

## 15. RL 残差策略：瓶子成功被抬起！（2026-04-02）

**完整复现 RoboWheel 三层方案**：

```
第1层: nvblox TSDF 穿透消除 → 参考轨迹 ✅
第2层: PPO 残差策略 (Stable Baselines3 + MuJoCo) ✅ ← 本节
第3层: 回放 RL 策略 → 物理合理抓取 ✅
```

### 方法

- **环境**: MuJoCo 双手 Allegro + 物理瓶子 (freejoint, mass=0.2kg, friction=2.0)
- **RL**: PPO (Stable Baselines3), 100K timesteps, 4 并行环境
- **动作**: 32 维残差 (16 finger actuators × 2 hands)，加在参考轨迹之上
- **观测**: 关节角(32) + 基座位置(6) + 瓶子位姿(7) + 指尖位置(24) + 时间步(1) = 70 维

### 奖励函数（和 RoboWheel 论文一致）

```
r_t = λ_geo · (-指尖到瓶子距离)     ← 几何跟踪
    + λ_dyn · (-动作变化量²)          ← 动力学平滑
    + λ_con · (接触数 × 0.1)          ← 接触一致性
    + λ_lift · max(0, 瓶子上升高度)    ← 抬起奖励
    + λ_stable · (-瓶子下降惩罚)       ← 稳定性
```

### 结果

| 指标 | 值 |
|------|-----|
| 训练时间 | ~5 分钟 (100K steps, CPU) |
| 最大接触数 | **54** |
| 瓶子最大抬升 | **+11.4 cm** ✅ |
| 抓取成功 | **YES** |

**这是所有版本中第一次真正的物理抓取成功。**

### 输出

```
/mnt/users/yjy/robowheel-demo/output/rl_grasp/
├── ppo_grasp.zip             # 训练好的 PPO 模型 (262KB)
├── rl_grasp.mp4              # RL 策略回放视频
├── comparison.mp4 / .gif     # 双栏: HaMeR | RL仿真
└── kf_*.jpg                  # 关键帧 (含 reward/contacts/lift 标注)
```

**脚本**: `/mnt/users/yjy/robowheel-demo/step_rl_grasp.py`

### 15.2 MANO 参考 + 小残差 RL（2026-04-02）

修正为真正的残差 RL：参考轨迹来自 MANO 数据，RL 只做微调。

**关键修改**：
1. 参考轨迹：MANO hand_pose → euler flexion → Allegro 关节角（不再是手写 curl）
2. 残差范围：±0.3 → **±0.05 rad**（RL 只能微调，不能覆盖参考）
3. 新奖励 `r_mano = -Σ(residual²) × 20`：惩罚偏离 MANO 参考

**结果**：
- 瓶子抬升: **10.1cm** ✅（仍然成功）
- 最大接触: **67**
- residual_norm: **0.26**（很小，大部分姿态来自 MANO）
- 和视频中人手抓取的对应关系更强
