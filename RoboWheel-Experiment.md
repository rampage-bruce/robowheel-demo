# RoboWheel 实验验证记录

> 基于 [RoboWheel-Survey.md](RoboWheel-Survey.md) 的调研，对 RoboWheel 核心链路进行逐步验证
> 日期：2026-03-31 ~ 2026-04-09
> 服务器：192.168.77.25, 8× NVIDIA RTX A5000 (24GB)

---

## 目录

- [1. HaMeR 手部 3D 重建](#1-hamer-手部-3d-重建验证)
- [2. HOI 3D 场景重建](#2-hoi-3d-场景重建)
- [3. 灵巧手重定向验证](#3-灵巧手重定向验证)
- [4. SPIDER 物理重定向](#4-spider-物理重定向验证)
- [5. SDF 穿透消除](#5-sdf-穿透消除)
- [6. 双手抓取 + nvblox Binary Search](#6-双手抓取--nvblox-binary-search-穿透修正)
- [7. MuJoCo 物理仿真交互](#7-mujoco-物理仿真交互)
- [8. RL 残差策略](#8-rl-残差策略)
- [9. Arti-MANO 1:1 映射](#9-arti-mano-11-mano-映射灵巧手)
- [10. SPIDER IK + Arti-MANO](#10-spider-ik--arti-mano)
- [11. ManipTrans 全控制 RL](#11-maniptrans-全控制-rl)
- [12. FoundationPose 物体追踪](#12-foundationpose-物体-6dof-追踪)
- [13. 最终统一管线](#13-最终统一管线foundationpose--arti-mano--ppo)
- [14. Hunyuan3D-2 物体重建](#14-hunyuan3d-2-物体重建)
- [15. Docker MJWP 尝试](#15-docker-mjwp-尝试)
- [附录：全版本对比总结](#附录全版本对比总结)

---

## 1. HaMeR 手部 3D 重建验证

使用 RoboWheel 论文的核心组件 **HaMeR**（Hand Mesh Recovery, ViT-H backbone）对随意视频进行手部姿态提取。

### 1.1 实验配置

| 项目 | 详情 |
|------|------|
| Conda 环境 | `hawor` (PyTorch 2.3.0+cu121, PyTorch3D 0.7.6) |
| 手部检测 | MediaPipe Hand Landmarker (CPU, 轻量) |
| 手部重建 | **HaMeR** (ViT-H backbone, ~2.6GB GPU 显存) |
| 测试视频 | `pick_bottle` — 人手抓取水瓶（15秒，640×360，30fps） |
| 测试帧数 | 151 帧 |

### 1.2 Pipeline

```
原始视频帧 (pick_bottle)
    ↓ MediaPipe Hand Landmarker (CPU)
手部 2D 边界框 + 左右手分类
    ↓ HaMeR (GPU, ViT-H)
MANO 3D 手部参数:
  - hand_pose (15 joints × 3×3 rot): 手指关节旋转
  - betas (10D): 手部形状
  - global_orient: 全局朝向
  - cam_t (3D): 相机空间位移
  - pred_vertices (778 顶点): 3D mesh
    ↓ PyRender 渲染
Overlay 图 + 侧视图 + OBJ mesh
```

### 1.3 结果

| 验证项 | 结果 |
|--------|------|
| 手部检测（视频→2D bbox） | ✅ 所有帧检测到手部 |
| 手部 3D 重建（2D→MANO） | ✅ 778 顶点 mesh + MANO 参数 |
| 渲染验证（mesh overlay） | ✅ 3D mesh 准确叠加到真实手部位置 |
| 左右手分类 | ✅ 正确区分 |
| 完整视频处理 | ✅ **151 帧，261 只手全部重建成功** |

### 1.4 实验结果

> **运行脚本**: `hamer/run_hamer_mp.py`
>
> ```bash
> cd /mnt/users/yjy/robowheel-demo/hamer
> CUDA_VISIBLE_DEVICES=5 python run_hamer_mp.py \
>   --img_folder ../test_videos/pick_bottle_frames \
>   --out_folder ../output/pick_bottle_video \
>   --batch_size=2 --side_view --save_mesh
> ```

![HaMeR 手部重建 151 帧](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/01_hamer_demo.gif)

**输出目录**: `output/pick_bottle_video/`
- `mano_results.json` — 261 个手部 MANO 参数
- `hamer_overlay.mp4` — overlay 视频
- `hamer_demo.gif` — overlay 动图

---

## 2. HOI 3D 场景重建

将 HaMeR 手部重建结果放入 3D 场景（手 + 瓶子 + 桌子），验证"视频 → 3D 场景"的可行性。

### 实验结果

> **运行脚本**: `hoi_sim_demo.py`
>
> ```bash
> MUJOCO_GL=egl python hoi_sim_demo.py
> ```

![HOI 3D 场景](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/02_hoi_demo.gif)

**输出目录**: `output/hoi_demo/`

---

## 3. 灵巧手重定向验证

在 HaMeR 手部重建基础上，验证 MANO → 灵巧手仿真的完整链路。测试了多种灵巧手模型。

### 3.1 Shadow Hand v2

**方法**: MANO 15 关节 → Shadow Hand E3M5 (20 actuator) 映射

**关节映射**:
| MANO 关节 | Shadow Actuator | 映射方式 |
|-----------|----------------|---------|
| global_orient → euler XZ | WRJ1, WRJ2 | 腕部弯曲/偏转 |
| thumb flex/spread | THJ5~THJ1 | 拇指旋转/对掌/弯曲 |
| index flex/spread | FFJ4, FFJ3, FFJ0 | 食指展开/MCP/PIP+DIP |
| middle, ring, pinky | 同上模式 | MFJ, RFJ, LFJ |

**问题**: 前臂体积过大遮挡手指

> **运行脚本**: `step_dexterous_v2.py`
>
> ```bash
> MUJOCO_GL=egl python step_dexterous_v2.py
> ```

![Shadow Hand v2](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/03_shadow_v2.gif)

**输出目录**: `output/dexterous_v2/`

### 3.2 Shadow Hand v3（碰撞修复）

修复碰撞检测，添加 3 阶段轨迹（approach → close → lift）。

> **运行脚本**: `step_dexterous_v3.py`
>
> ```bash
> MUJOCO_GL=egl python step_dexterous_v3.py
> ```

![Shadow Hand v3](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/04_shadow_v3.gif)

**输出目录**: `output/dexterous_v3/`

### 3.3 多灵巧手对比测试

在同一个 MANO 轨迹上测试 3 种灵巧手模型。

| 手模型 | 来源 | 关节 | 效果 |
|--------|------|------|------|
| **Allegro V3** | Wonik Robotics | 16 | **最佳** — 手指动态清晰 |
| LEAP Hand | CMU | 16 | 挂载位置错误 |
| Shadow DEXee | Shadow Robot | 12 | 只有 3 个 actuator 正确 |

> **运行脚本**: `step_multi_hand.py`
>
> ```bash
> MUJOCO_GL=egl python step_multi_hand.py
> ```

![多灵巧手对比](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/05_multi_hand.gif)

**输出目录**: `output/multi_hand/`

### 3.4 Allegro v2 比例修正

修正瓶子尺寸为真实比例（直径 6.6cm, 高度 20cm）。

> **运行脚本**: `step_multi_hand.py`（修改瓶子尺寸参数）

![Allegro v2 比例修正](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/06_allegro_v2.gif)

**输出目录**: `output/multi_hand_v2/`

### 3.5 灵巧手迭代总结

| 版本 | 手模型 | 手指可见 | 碰撞 | 动作序列 | 抓取成功 |
|------|--------|---------|------|---------|---------|
| v2 | Shadow | 被前臂遮挡 | 穿模 | 无阶段 | 否 |
| v3 | Shadow | 部分可见 | 正常 | 3 阶段 | 否 |
| multi | Allegro/LEAP/DEXee | Allegro 最佳 | 正常 | 5 阶段 | 否 |
| **Allegro v2** | **Allegro** | **完全可见** | **正常** | **5 阶段** | **否（碰翻了）** |

**为什么需要 RL 精化**：开环映射暴露了 3 个根本问题：
1. **手-物尺度不匹配** — 开环映射不感知物体大小
2. **接近路径缺失** — MANO 只有姿态没有路径
3. **缺少接触力闭环** — 不知道手指是否碰到物体

---

## 4. SPIDER 物理重定向验证

使用 [SPIDER](https://github.com/facebookresearch/spider)（Facebook Research）实现 MuJoCo 原生的物理重定向。

### 4.1 示例数据验证 ✅

SPIDER IK + MJWP 在示例数据（fair_fre/pour, Allegro bimanual）上成功运行。

### 4.2 自有数据接入

1. ✅ `convert_mano_to_spider.py`：HaMeR MANO → SPIDER `trajectory_keypoints.npz`
2. ✅ 坐标变换：MANO 相机空间 → SPIDER 世界空间
3. ✅ `ik_fast.py`：150 帧 IK 运动学重定向
4. ❌ `run_mjwp.py`：MJWP 物理优化失败 — **Warp CUDA 内核编译 segfault**

**MJWP 失败根因**: NVIDIA Driver 570.195.03（CUDA 13.0）与 Warp 1.12/1.13 的 NVRTC 编译器不兼容。

---

## 5. SDF 穿透消除

### 5.1 trimesh SDF 精化

SPIDER IK + trimesh SDF 穿透消除。

```
SPIDER IK 运动学轨迹 (147帧)
    ↓ 逐帧 SDF 穿透检查 (指尖到瓶子 mesh 符号距离)
    ↓ scipy L-BFGS-B 优化 (调整手指关节角)
    ↓ 时序平滑 → MuJoCo 回放渲染
```

| 指标 | 值 |
|------|-----|
| 穿透修复数 | **80 处** |
| 穿透阈值 | 2mm |
| 最大关节调整 | 0.046 rad (~2.6°) |

> **运行脚本**: `step_sdf_refine.py`
>
> ```bash
> MUJOCO_GL=egl python step_sdf_refine.py
> ```

![trimesh SDF 精化 (三栏: HaMeR - IK原始 - SDF精化)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/07_sdf_refined.gif)

**Hunyuan3D-2 瓶子 mesh 版本**:

![Hunyuan3D-2 mesh SDF 精化](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/07b_sdf_hunyuan.gif)

**输出目录**: `output/sdf_refined/`

### 5.2 侧面接近轨迹优化

添加 4 阶段渐进 curl + 弧线接近。

| 阶段 | 时间 | 手指动作 | 基座运动 |
|------|------|---------|---------|
| APPROACH | 0-30% | 保持 IK 原始 | 弧线接近物体 |
| CLOSE | 30-60% | 渐进 curl +0.4 rad | 闭合间隙 |
| GRASP | 60-80% | 强 curl +0.7 rad | 保持 |
| LIFT | 80-100% | 保持 curl | Z+5cm 抬起 |

> **运行脚本**: `step_approach_refine.py`
>
> ```bash
> MUJOCO_GL=egl python step_approach_refine.py
> ```

![接近轨迹优化 (三栏: HaMeR - IK原始 - 接近精化)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/08_approach_refined.gif)

**输出目录**: `output/approach_refined/`

### 5.3 nvblox GPU SDF 精化

用 NVIDIA nvblox 替换 trimesh 作为 GPU SDF 后端。

| | trimesh (之前) | nvblox (现在) |
|--|-------------|------------|
| 运行设备 | CPU | **GPU (CUDA)** |
| 穿透检测数 | 80 | **253** |
| 穿透阈值 | 2mm | **0.5mm** |
| SDF 构建 | mesh nearest | **TSDF 12 视角深度渲染** |
| 批量查询 | 逐点 | **GPU 批量** |

> **运行脚本**: `step_sdf_nvblox.py`
>
> ```bash
> MUJOCO_GL=egl python step_sdf_nvblox.py
> ```

![nvblox GPU SDF 精化 (三栏: HaMeR - IK原始 - nvblox精化)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/09_sdf_nvblox.gif)

**输出目录**: `output/sdf_nvblox/`

### 5.4 统一管线：分阶段轨迹 + nvblox SDF

合并所有优化：5 阶段抓取轨迹 + nvblox GPU SDF + 直接推出策略。

**5 阶段**: REACH (40帧) → APPROACH → CLOSE → GRASP → LIFT

> **运行脚本**: `step_unified_grasp.py`
>
> ```bash
> MUJOCO_GL=egl python step_unified_grasp.py
> ```

![统一管线 (三栏: HaMeR - IK原始 - 统一精化)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/10_unified_grasp.gif)

**输出目录**: `output/unified_grasp/`

---

## 6. 双手抓取 + nvblox Binary Search 穿透修正

左右手 MANO → 双 Allegro Hand → 5 阶段轨迹 → nvblox 二分搜索穿透修正。

```
HaMeR MANO (151帧右手 + 110帧左手)
    ↓ build_bimanual_scene()
MuJoCo: 2× Allegro Hand + 蓝色圆柱瓶 + 桌子
    ↓ 5-stage trajectory
REACH(40帧) → APPROACH → CLOSE → GRASP → LIFT
    ↓ nvblox binary search (6次二分 = 1/64 精度)
穿透修正 → MuJoCo 渲染
```

| 指标 | 值 |
|------|-----|
| 双手帧数 | 150 |
| 模型 | 2× Allegro Hand = 38 actuators |
| 穿透检测 | **1 处**（之前单手 131 处） |

**迭代过程**:
1. 初始版本：穿透 131 处 → binary search 修正
2. 手朝向修正："朝下"改为"水平伸手+掌心相对"
3. MANO global_orient 自动驱动（缩放 15%）
4. 稳定版：固定水平朝向 + staged X 轴接近

> **运行脚本**: `step_bimanual_grasp.py`
>
> ```bash
> MUJOCO_GL=egl python step_bimanual_grasp.py
> ```

![双手抓取 + nvblox Binary Search](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/11_bimanual_grasp.gif)

**输出目录**: `output/bimanual_grasp/`

---

## 7. MuJoCo 物理仿真交互

用 MuJoCo 原生接触物理替代外部 SDF 检查。

| | 之前 (所有版本) | 现在 (physics_grasp) |
|--|-------------|---------------------|
| 瓶子 | static/visual-only | **freejoint + 碰撞 + 质量 + 摩擦** |
| 穿透检测 | nvblox/trimesh 外部 SDF | **MuJoCo 内置接触求解器** |
| 物理 | 无 | **重力 + 摩擦 + 接触力** |
| 接触数 | 0 | **最高 15 个同时接触** |

**关键参数**: mass=0.25kg, friction=[1.5,0.01,0.001], impratio=10, cone=ELLIPTIC

**结果**: 260 帧稳定，**手指不再穿透瓶子**。瓶子未抬起（握力 < 重力）。

> **运行脚本**: `step_physics_grasp.py`
>
> ```bash
> MUJOCO_GL=egl python step_physics_grasp.py
> ```

![MuJoCo 物理仿真 (双栏: HaMeR - 物理仿真)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/12_physics_grasp.gif)

**输出目录**: `output/physics_grasp/`

---

## 8. RL 残差策略

**复现 RoboWheel 三层方案**:
```
第1层: nvblox TSDF 穿透消除 → 参考轨迹
第2层: PPO 残差策略 (Stable Baselines3 + MuJoCo)
第3层: 回放 RL 策略 → 物理合理抓取
```

### 8.1 V1 从零 RL：瓶子成功抬起！

- 环境: MuJoCo 双手 Allegro + 物理瓶子 (freejoint, mass=0.2kg, friction=2.0)
- RL: PPO, 100K~500K timesteps, 4~16 并行环境
- 动作: 32 维残差加在**手写 curl 参考**之上
- 观测: 关节角(32) + 基座(6) + 瓶子(7) + 指尖(24) + 时间(1) = 70 维

**奖励函数（和 RoboWheel 论文一致）**:
```
r = λ_geo·(-指尖到瓶距离) + λ_dyn·(-动作变化²)
  + λ_con·(接触数×0.1)    + λ_lift·max(0, 抬升高度)
  + λ_stable·(-下降惩罚)
```

| 指标 | 100K 基线 | 500K 优化 |
|------|---------|---------|
| 最大接触数 | 54 | **75** |
| 瓶子最大抬升 | 10.1cm | **11.6cm** ✅ |
| value_loss | 3400 | **0.03** (VecNormalize) |
| explained_var | 0.65 | **0.96** |

**这是所有版本中第一次真正的物理抓取成功。**

> **运行脚本**: `step_rl_grasp.py`
>
> ```bash
> MUJOCO_GL=egl python step_rl_grasp.py
> ```

![V1 RL 抓取 (双栏: HaMeR - RL仿真)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/13_rl_grasp.gif)

**输出目录**: `output/rl_grasp/`

### 8.2 V1 MANO 参考 + 小残差 RL

参考轨迹改为 **MANO 数据**，RL 只做 ±0.05rad 微调。

**关键修改**:
1. 参考：MANO hand_pose → euler flexion → Allegro 关节角
2. 残差范围：±0.3 → **±0.05 rad**
3. 新奖励 `r_mano = -Σ(residual²) × 20`

**结果**: 瓶子抬升 **10.1cm** ✅, 最大接触 **67**, residual_norm **0.26**

### 8.3 V1.5 混合方案（最终最佳抬升）

```
base 运动:   V1 staged trajectory (approach → hold → lift)   ← 物理稳定
手指弯曲:   V2 MANO hand_pose → Allegro (每帧)               ← 视觉正确
手朝向:     固定水平 (右Z=180°, 左Y=15°)                     ← 不翻倒
RL 残差:    ±0.05rad 32维 (仅手指)                            ← 微调物理
```

| 指标 | V1 (手写curl) | V2 (纯MANO) | **V1.5 (混合)** |
|------|----------|------------|---------------|
| 手指来源 | 手写curl | MANO | **MANO** ✅ |
| base来源 | staged | MANO | **staged** |
| 瓶子抬升 | 11.6cm | 飞走 | **9.9cm** ✅ |
| 物理稳定 | ✅ 200帧 | ❌ 16帧 | **✅ 200帧** |
| max contacts | 67 | - | **83** |

**V1.5 是唯一同时满足"MANO 手指 + 瓶子抬起 + 200帧稳定"的方案。**

> **运行脚本**: `step_rl_v15.py`
>
> ```bash
> MUJOCO_GL=egl python step_rl_v15.py
> ```

![V1.5 混合方案 (双栏: HaMeR - V1.5 仿真)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/14_rl_v15.gif)

**输出目录**: `output/rl_v15/`

### 8.4 V2 纯 MANO 驱动

100% MANO 驱动所有 DOF，RL 只做物理修正。

**问题**: wrist euler 太大 → 手翻倒；MANO finger angles 超出 Allegro 安全范围 → 弹飞瓶子

**结果**: 手指形状最接近视频，但 40 帧后瓶子飞走。

> **运行脚本**: `step_rl_v2.py`
>
> ```bash
> MUJOCO_GL=egl python step_rl_v2.py
> ```

![V2 纯 MANO (双栏: HaMeR - V2 仿真)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/15_rl_v2.gif)

**输出目录**: `output/rl_v2/`

---

## 9. Arti-MANO: 1:1 MANO 映射灵巧手

SPIDER 自带 **Arti-MANO**: MANO 手模型的精确 MuJoCo 实现。

| | Allegro (之前) | Arti-MANO (现在) |
|--|-------------|-----------------|
| 手指数 | 4（无小指） | **5（含小指）** |
| 总手指 DOF | 16 | **22** |
| 映射方式 | euler X 近似 | **1:1 直接映射** |
| 视觉 | 机器人方块手 | **真实人手 mesh** |
| 信息丢失 | 严重（3DOF→1DOF） | **零** |

### MANO → Arti-MANO 映射

```python
MANO joint (3×3 rot) → euler xyz → Arti-MANO:
  Index/Middle/Ring/Pinky: euler[2]→spread, euler[0]→flex × 3 joints
  Thumb: euler[0,1,2] → 3+2+1 DOF
```

**结果**:
- **Frame 0 手形完全正确**: 真实人手形状、5 指、正确弯曲
- 后续帧手飞走：**核心瓶颈** — MANO 没有提供手的绝对世界位置

> **运行脚本**: `step_rl_manohand.py`
>
> ```bash
> MUJOCO_GL=egl python step_rl_manohand.py
> ```

![Arti-MANO 1:1 映射 (双栏: HaMeR - Arti-MANO 仿真)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/16_rl_manohand.gif)

**输出目录**: `output/rl_manohand/`

---

## 10. SPIDER IK + Arti-MANO

用 SPIDER IK 的 base position 驱动 Arti-MANO 世界坐标，手指用 MANO 1:1 映射。

**坐标对齐公式**: `hand_pos_ours = (spider_hand - spider_obj) × 0.3 + our_bottle_pos`

**结果**: Frame 0 双手 Arti-MANO 在瓶子两侧（视觉最佳），后续帧 SPIDER IK delta 偏移过大。

> **运行脚本**: `step_rl_final.py`
>
> ```bash
> MUJOCO_GL=egl python step_rl_final.py
> ```

![SPIDER IK + Arti-MANO (双栏: HaMeR - SPIDER+Arti-MANO)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/18_rl_final.gif)

**输出目录**: `output/rl_final/`

---

## 11. ManipTrans 全控制 RL

严格按 RoboWheel/ManipTrans 方案：RL 输出**完整控制量**，参考轨迹只在 reward 中。

```
观测: [当前状态(28), 参考目标(28), 瓶子(7), 接触(1), 时间(1)] = 65维
动作: 完整 actuator 控制 28维 (6 base + 22 fingers)
```

### 训练量对比

| | ManipTrans 论文 | 我们的实现 | 差距 |
|--|---------------|-----------|------|
| 并行环境 | 4096 (GPU) | 16 (CPU) | **256×** |
| 训练步数 | 数百万 | 300K | **10×+** |
| 总样本量 | ~10 亿 | ~5 百万 | **200×** |

**结论**: 代码框架正确，但 **训练量差距 200 倍无法通过改代码弥补**。

> **运行脚本**: `step_maniptrans.py`
>
> ```bash
> MUJOCO_GL=egl python step_maniptrans.py
> ```

![ManipTrans 全控制 RL (双栏: HaMeR - ManipTrans)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/17_maniptrans.gif)

**输出目录**: `output/maniptrans/`

---

## 12. FoundationPose 物体 6DoF 追踪

**突破**: FoundationPose 在宿主机上成功运行，追踪了全部 151 帧的瓶子 6DoF 位姿。

### 解决过程

1. ✅ nvdiffrast 0.4.0 + 系统 CUDA 12.8 编译 + PyTorch 2.1+cu121
2. ✅ Warp segfault → **OpenCV 替换** erode_depth / bilateral_filter_depth（修改 `FoundationPose/Utils.py`）
3. ✅ Qt display crash → `--debug 0` headless
4. ✅ mycpp import → 创建 `__init__.py`
5. ✅ `RasterizeCudaContext` → `RasterizeGLContext`（修改 `FoundationPose/estimater.py`）

### 输出

```
output/bottle_6dof_poses.npz
  poses: (151, 4, 4) — 每帧瓶子 6DoF 位姿
  X: [-0.03, +0.12]m, Y: [-0.01, +0.10]m, Z: [0.46, 0.54]m
```

### 关键发现: cam_t 校准

**MANO cam_t[2]=35.5 不是米数**（HaMeR 内部归一化值，差 68 倍）:

```python
x_ndc = cam_t[0] / cam_t[2]  # 归一化到 NDC
y_ndc = cam_t[1] / cam_t[2]
hand_cam = [x_ndc * FPose_depth, y_ndc * FPose_depth, FPose_depth]
```

**运行命令**:
```bash
conda activate fpose
cd FoundationPose
python run_demo.py --mesh_file demo_data/pick_bottle/mesh.obj \
  --test_scene_dir demo_data/pick_bottle --est_refine_iter 5 --track_refine_iter 2 --debug 0
```

---

## 13. 最终统一管线：FoundationPose + Arti-MANO + PPO

### 完整 Pipeline

```
视频
  ↓ HaMeR + MediaPipe
MANO 手部参数 (151帧双手)
  ↓ FoundationPose
瓶子 6DoF (151帧, 相机坐标)
  ↓ cam_t NDC × FPose_depth + lateral_offset
手世界坐标 (相对瓶子)
  ↓ Arti-MANO 1:1 手指映射
22 DOF 手指关节角 (每帧)
  ↓ PPO ±0.08rad 残差 RL
物理微调
  ↓ MuJoCo 仿真 → 渲染
```

### 坐标转换

```python
# 手-瓶偏移 (相机坐标 → 世界坐标)
hand_offset_cam = hand_cam - bottle_cam
w[0] = offset[0]         # X: 左右
w[1] = -offset[2] * 0.3  # Y: 深度差 → 前后
w[2] = -offset[1]        # Z: cam Y(下) → world Z(上)

# 侧向偏移
LATERAL_OFFSET_R = [+0.08, 0, 0.02]  # 右手 +8cm X
LATERAL_OFFSET_L = [-0.08, 0, 0.02]  # 左手 -8cm X
```

### 结果

**Frame 0** — 整个项目视觉效果最好的一帧:
- 双手 Arti-MANO 真实人手 mesh，5 指自然弯曲
- 瓶子在双手之间，手指不穿透

**待优化**: 后续帧手偏离瓶子（base 刚度不足 / reward 不平衡）

> **运行脚本**: `step_final_unified.py`
>
> ```bash
> MUJOCO_GL=egl python step_final_unified.py
> ```

![最终统一管线 (双栏: HaMeR - FoundationPose+Arti-MANO+PPO)](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/19_final_unified.gif)

**Frame 0 关键帧**:

![Frame 0: 双手 Arti-MANO 在瓶子两侧](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/final_frame0.jpg)

**Frame 32 关键帧**（手偏移问题）:

![Frame 32: 左手偏离瓶子](https://raw.githubusercontent.com/12vv/robowheel-demo/main/docs/images/final_frame32.jpg)

**输出目录**: `output/final_unified/`

---

## 14. Hunyuan3D-2 物体重建

从视频截帧用 Hunyuan3D-2 AI 生成水瓶 3D mesh。

| 指标 | 值 |
|------|-----|
| 模型 | tencent/Hunyuan3D-2 (DiT Flow Matching) |
| 原始 mesh | 172,476 顶点, 344,944 面 |
| 简化后 | 2,502 顶点, 5,000 面 |
| 缩放尺寸 | 8.8cm × 20cm × 12.5cm |
| GPU 显存 | ~16GB, ~8 分钟 |

**输出**: `output/bottle_hunyuan3d.obj`, `output/bottle_hunyuan3d_scaled.obj`

---

## 15. Docker MJWP 尝试

尝试通过 Docker 绕过 Warp segfault。

| 问题 | 详情 |
|------|------|
| Docker 权限 | ✅ `sg docker` 可用 |
| GPU 访问 | ✅ nvidia-smi 正常 |
| Docker 网络 | ❌ **无法访问外网** |
| CUDA 兼容性 | ❌ host driver 570=CUDA 13.0 不兼容 |

**结论**: Docker 路线被网络和 CUDA 版本双重阻塞。

---

## 附录：全版本对比总结

### 方案迭代全览

| # | 方案 | 手模型 | 手指来源 | base 来源 | 手形 | 抬升 | 稳定帧 | 脚本 |
|---|------|--------|---------|----------|------|------|-------|------|
| 1 | HaMeR 重建 | MANO mesh | - | - | ✅ | - | 151 | `hamer/run_hamer_mp.py` |
| 2 | Shadow v2 | Shadow 20DOF | euler | 手写 | ⚠️ | 0 | 150 | `step_dexterous_v2.py` |
| 3 | Shadow v3 | Shadow 20DOF | euler | 3阶段 | ⚠️ | 0 | 150 | `step_dexterous_v3.py` |
| 4 | Allegro | Allegro 16DOF | euler X | 5阶段 | ⚠️ | 0 | 150 | `step_multi_hand.py` |
| 5 | trimesh SDF | Allegro | IK | SPIDER | ⚠️ | 0 | 147 | `step_sdf_refine.py` |
| 6 | nvblox SDF | Allegro | IK | SPIDER | ⚠️ | 0 | 147 | `step_sdf_nvblox.py` |
| 7 | 统一轨迹 | Allegro | IK+curl | 5阶段 | ⚠️ | 0 | 187 | `step_unified_grasp.py` |
| 8 | 双手 nvblox | 2×Allegro | IK+curl | 5阶段 | ⚠️ | 0 | 150 | `step_bimanual_grasp.py` |
| 9 | MuJoCo 物理 | 2×Allegro | staged | 5阶段 | ⚠️ | 0 | 260 | `step_physics_grasp.py` |
| 10 | **V1 RL** | 2×Allegro | **curl+PPO** | staged | ❌ | **11.6cm** ✅ | 200 | `step_rl_grasp.py` |
| 11 | V1 MANO残差 | 2×Allegro | MANO+PPO | staged | ⚠️ | **10.1cm** ✅ | 200 | `step_rl_grasp.py` |
| 12 | **V1.5 混合** | 2×Allegro | **MANO+PPO** | **staged** | ⚠️ | **9.9cm** ✅ | **200** | `step_rl_v15.py` |
| 13 | V2 纯MANO | 2×Allegro | MANO+PPO | MANO | ✅ | 飞走 ❌ | 16 | `step_rl_v2.py` |
| 14 | Arti-MANO | AM 22DOF | MANO 1:1 | MANO | ✅ F0 | 0 | 1 | `step_rl_manohand.py` |
| 15 | SPIDER+AM | AM | MANO 1:1 | SPIDER | ✅ F0 | 0 | 1 | `step_rl_final.py` |
| 16 | ManipTrans | AM | RL 全控制 | RL | ❌ | 0 | 80 | `step_maniptrans.py` |
| 17 | **统一管线** | **AM** | **MANO+PPO** | **FPose** | **✅ F0** | 0 | 80 | `step_final_unified.py` |

### 最佳成果

| 指标 | 最佳方案 | 值 |
|------|---------|-----|
| **最佳抬升** | V1.5 (Allegro + staged + MANO + PPO) | **9.9cm, 200帧, 83 contacts** |
| **最佳手形** | 统一管线 (Arti-MANO + FoundationPose) | **Frame 0 完美人手** |
| **最佳穿透** | MuJoCo 原生物理 | **零穿透, 15 contacts** |
| **最佳 SDF** | nvblox GPU TSDF | **253 处检测, binary search** |
| **物体追踪** | FoundationPose | **151 帧 6DoF** |

### 工具栈

| 工具 | 用途 | 状态 |
|------|------|------|
| [HaMeR](https://github.com/geopavlakos/hamer) | 手部 MANO 3D 重建 | ✅ |
| [MediaPipe](https://github.com/google/mediapipe) | 手部 2D 检测 | ✅ |
| [FoundationPose](https://github.com/NVlabs/FoundationPose) | 物体 6DoF 追踪 | ✅ |
| [nvblox](https://github.com/nvidia-isaac/nvblox) | GPU TSDF/SDF | ✅ |
| [SPIDER](https://github.com/facebookresearch/spider) | IK 运动学重定向 | ✅ |
| [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) | PPO RL | ✅ |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 物理仿真 | ✅ |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | 可微分渲染 | ✅ |
| Arti-MANO (SPIDER 内置) | MANO 1:1 仿真手 | ✅ |

### 待完成

1. **后续帧手位置跟踪** — Frame 0 后手偏离瓶子
2. **更多 RL 训练量** — ManipTrans 需要 GPU 并行
3. **物体真实重建** — Hunyuan3D2 接入 FoundationPose
4. **双手全帧统一** — 目前最佳是单帧（Frame 0）
5. **Warp/CUDA 兼容** — 等 Warp 更新或降级 driver

---

*最后更新: 2026-04-09*
