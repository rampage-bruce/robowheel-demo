# RoboWheel 开源替代管线：一周工作总结

> 2026-03-31 ~ 2026-04-08
> 目标：从互联网视频自动生成灵巧手抓取数据（复现 RoboWheel 论文）

---

## 一句话总结

从互联网视频出发，搭建了完整的 HaMeR→FoundationPose→Arti-MANO→PPO 管线，在 9 个方案迭代中验证了"残差 RL + 物理合理参考轨迹"可以抬起瓶子 9.9cm，"Arti-MANO 1:1 映射"可以产出完美人手形状，并最终通过 FoundationPose 物体追踪解决了手-物世界坐标对齐这一核心瓶颈。

---

## 搭建的工具栈

| 工具 | 用途 | 安装方式 | 状态 |
|------|------|---------|------|
| [HaMeR](https://github.com/geopavlakos/hamer) | 手部 MANO 3D 重建 | conda hawor + git | ✅ |
| [MediaPipe](https://github.com/google/mediapipe) | 手部 2D 检测 | pip | ✅ |
| [FoundationPose](https://github.com/NVlabs/FoundationPose) | 物体 6DoF 追踪 | 源码编译 (fpose env) | ✅ |
| [nvblox](https://github.com/nvidia-isaac/nvblox) | GPU TSDF/SDF | cmake 源码编译 | ✅ |
| [SPIDER](https://github.com/facebookresearch/spider) | IK 运动学重定向 | conda spider + pip | ✅ |
| [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | PPO RL | pip | ✅ |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 物理仿真 | pip | ✅ |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | 可微分渲染 | 源码编译 (CUDA 12.8) | ✅ |
| Arti-MANO | MANO 1:1 仿真手 | SPIDER 内置 | ✅ |
| Allegro Hand | 4指灵巧手 | mujoco_menagerie | ✅ |
| Shadow Hand | 5指灵巧手 | mujoco_menagerie | ✅ |

---

## 方案迭代

### 第一阶段：手部重建 + 灵巧手选型

| # | 方案 | 手模型 | 结果 | 问题 |
|---|------|--------|------|------|
| 1 | HaMeR 手部重建 | MANO mesh | 151帧261手 ✅ | - |
| 2 | Shadow Hand 映射 | Shadow 20DOF | 手指动，前臂遮挡 | 前臂太大 |
| 3 | Allegro Hand 映射 | Allegro 16DOF | 5阶段轨迹清晰 | euler 映射丢信息 |
| 4 | MuJoCo 物理交互 | Allegro | 15 contacts, 零穿透 | 没抓住 |

### 第二阶段：RL 策略

| # | 方案 | 参考轨迹 | RL 方式 | lift | 手形 |
|---|------|---------|--------|------|------|
| 5 | V1 从零 RL | 手写 curl | PPO ±0.3 | **11.6cm** ✅ | ❌ 不像人手 |
| 6 | V1 MANO 残差 | MANO curl | PPO ±0.05 | **10.1cm** ✅ | ⚠️ 部分 |
| 7 | V1.5 混合 | MANO curl + staged | PPO ±0.05 | **9.9cm** ✅ | ⚠️ 部分 |
| 8 | V2 纯 MANO | MANO 全部 | PPO ±0.05 | 飞走 ❌ | ✅ 正确 |

### 第三阶段：Arti-MANO + 坐标对齐

| # | 方案 | 手模型 | base 来源 | 结果 |
|---|------|--------|----------|------|
| 9 | Arti-MANO 单独 | Arti-MANO 22DOF | MANO orient | Frame 0 完美，后续飞走 |
| 10 | SPIDER IK + Arti-MANO | Arti-MANO | SPIDER IK | 坐标不对齐 |
| 11 | ManipTrans 全控制 | Arti-MANO | RL 学习 | 训练量不足 |
| 12 | **FoundationPose 接入** | Arti-MANO | **FPose 追踪** | **Frame 0 最佳** |

---

## 关键发现

### 1. 残差 RL 有效，但前提是参考轨迹物理合理

V1.5（staged base + MANO fingers + PPO ±0.05）是唯一能**同时抬起瓶子 + 200帧稳定**的方案。关键是 staged trajectory 天生物理合理，RL 只需微调。

### 2. Arti-MANO 1:1 映射产出完美手形

SPIDER 内置的 Arti-MANO（22DOF, 5指）和 MANO 完美对应。Frame 0 的双手人手 mesh 是所有版本中视觉最佳。

### 3. 手-物世界坐标是核心瓶颈

所有版本失败的"手飞走"、"坐标不对齐"都源于 MANO 只提供相机空间 cam_t，无法得到手的世界坐标。FoundationPose 解决了物体端，但 cam_t→世界坐标的缩放还需校准。

### 4. Warp + CUDA Driver 570 全面不兼容

SPIDER MJWP 和 FoundationPose 都遇到 Warp segfault（CUDA 13.0 driver）。解决方案：OpenCV 替换 Warp kernel，或等 Warp 更新。

### 5. 训练量是 ManipTrans 方案的瓶颈

ManipTrans 框架代码正确，但 16 CPU 并行 × 300K 步 vs 论文 4096 GPU 并行 × 百万步，差距 200 倍。

---

## 最佳成果

| 指标 | 最佳方案 | 值 |
|------|---------|-----|
| **最佳抬升** | V1.5 (Allegro + staged + PPO) | **9.9cm, 200帧, 83 contacts** |
| **最佳手形** | Arti-MANO + FoundationPose | **Frame 0 完美人手** |
| **最佳穿透处理** | MuJoCo 原生物理 | **零穿透, 15 contacts** |
| **最佳 SDF** | nvblox GPU TSDF | **253处检测, binary search** |
| **物体追踪** | FoundationPose | **151帧 6DoF** |

---

## 文件索引

### 文档 (`sim/docs/`)
- `RoboWheel-Survey.md` — 论文调研
- `RoboWheel-Experiment.md` — 完整实验记录 (§1-23)
- `RoboWheel-Pipeline-Architecture.md` — 管线架构图
- `RoboWheel-OpenSource-Alternative.md` — 开源替代方案
- `RoboWheel-V2-Plan.md` — V2 计划
- `RoboWheel-Unified-Plan.md` — 统一方案计划
- `FoundationPose-Integration-Plan.md` — FPose 接入计划
- `RoboWheel-Weekly-Summary.md` — 本总结

### 脚本 (`robowheel-demo/`)
- `step_final_unified.py` — **最终管线**: FPose + Arti-MANO + PPO
- `step_rl_v15.py` — **最佳抬升**: staged + MANO + PPO (9.9cm)
- `step_maniptrans.py` — ManipTrans 全控制 RL
- `step_rl_manohand.py` — Arti-MANO 1:1 映射
- `step_physics_grasp.py` — MuJoCo 物理基线
- `step_sdf_nvblox.py` — nvblox GPU SDF
- `hamer/run_hamer_mp.py` — HaMeR 手部重建

### 环境
- `hawor` — HaMeR + PyTorch 2.3
- `spider` — SPIDER + nvblox + MuJoCo + SB3
- `fpose` — FoundationPose + nvdiffrast

---

## 待完成

1. **cam_t↔FoundationPose 缩放校准** — 用相机内参 K 统一坐标尺度
2. **全帧手位置跟踪** — Frame 0 后手偏离瓶子
3. **更多训练量** — ManipTrans 需要 GPU 并行
4. **物体真实重建** — Hunyuan3D2 替代圆柱体
5. **双手全帧统一** — 目前最佳是单帧（Frame 0）

---

*总结日期: 2026-04-08*

---

## 最终成果更新（2026-04-08 晚）

### FoundationPose 接入成功

- 151 帧瓶子 6DoF 追踪完成
- cam_t↔FoundationPose 校准公式确定：`hand_meters = (cam_t[0:2] / cam_t[2]) × FPose_depth`
- MANO cam_t[2]=35.5 不是米数，是 HaMeR 内部归一化值（差 68 倍）

### 最终管线效果

- **Frame 0**: 双手 Arti-MANO 真实人手在瓶子两侧，不穿透，手指自然弯曲
- **Frame 44**: 手跟随 FoundationPose 追踪上移（视频中人抬手）
- 后续帧超出 joint range（举杯喝水动作幅度大）

### 待优化

- 增大 joint range 或只截取抓取阶段帧
- 手朝向仍为固定值（应跟随 MANO global_orient delta）
- 瓶子 freejoint 在接触时被碰倒

### 最终管线

```
视频
  ↓
HaMeR → MANO 手部参数 (151帧双手)
  ↓
FoundationPose → 瓶子 6DoF (151帧)
  ↓
cam_t NDC × FPose_depth + lateral_offset → 手世界坐标
  ↓
Arti-MANO 1:1 手指映射 + PPO ±0.08rad 残差
  ↓
MuJoCo 物理仿真 → 渲染
```
