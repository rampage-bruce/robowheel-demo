# RoboWheel V2: 完整复现计划

> 目标：关节角、手朝向、手位置全部来自 MANO 数据，RL 只做物理微调
> 日期：2026-04-02

---

## V1 的问题

```
V1 的参考轨迹组成:
  手指弯曲:  来自 MANO ✅
  手的位置:  手写 approach -6cm / lift +6cm ❌
  手的朝向:  硬编码 Z=π ❌
  base 运动: 手写 staged trajectory ❌

→ 只有 ~20% 来自 MANO，80% 是手写的
→ RL 在错误的参考上微调 → 关节不对
```

## V2 的方案（和 RoboWheel 一致）

```
V2 的参考轨迹组成:
  手指弯曲:  来自 MANO hand_pose ✅
  手的位置:  来自 MANO joints[0] (wrist) + cam_t ✅
  手的朝向:  来自 MANO global_orient ✅
  物体位置:  来自 MANO thumb/index 中点 ✅

→ 100% 来自 MANO 数据
→ RL 只做 ±0.05rad 残差物理修正
```

---

## 执行步骤

### Step 1: 构建完整 MANO 参考轨迹

**输入**: `mano_results.json` (151 帧右手 + 110 帧左手)

**输出**: 每帧完整的 Allegro 控制量
```
per_frame = {
    wrist_pos:     (3,)   ← MANO joints[0] + cam_t → 世界坐标
    wrist_orient:  (3,3)  ← MANO global_orient → 世界旋转矩阵
    finger_angles: (16,)  ← MANO hand_pose 15 joints → Allegro 16 actuators
    obj_pos:       (3,)   ← (thumb_tip + index_tip) / 2
}
```

**关键转换**:
- MANO 相机空间 → MuJoCo 世界空间（Y/Z 交换 + 偏移）
- MANO global_orient (3×3) → MuJoCo euler XYZ（base hinge 关节角）
- MANO hand_pose 旋转矩阵 → euler flexion → Allegro 关节角
- 双手: 右手直接映射，左手 X 轴镜像

### Step 2: MuJoCo 环境（全 MANO 驱动）

**场景**:
- 2× Allegro Hand (16 DOF + 6D base = 22 per hand)
- 瓶子 (freejoint, 碰撞, 物理)
- 桌面 + 地面

**base 关节**:
- 3 slide (XYZ 位置) → 由 MANO wrist_pos 驱动
- 3 hinge (RPY 旋转) → 由 MANO wrist_orient 驱动

**手指关节**:
- 16 position actuators → 由 MANO finger_angles 驱动

**区别 V1**: base 位置和旋转不再是手写的 staged trajectory，而是逐帧来自 MANO

### Step 3: PPO 残差 RL

**动作空间**: 44 维残差 (22 per hand × 2)
- 6D base 残差: ±0.01m (位置), ±0.05rad (旋转)
- 16D finger 残差: ±0.05rad

**观测空间**: 70 维 (同 V1)

**奖励函数** (和 RoboWheel 一致):
```
r = λ_track × (-Σ|q - q_ref|²)    ← MANO 跟踪 (权重最大!)
  + λ_geo   × (-指尖到瓶子距离)     ← 几何接近
  + λ_dyn   × (-Σ|a_t - a_{t-1}|²) ← 动力学平滑
  + λ_con   × (接触数)              ← 接触奖励
  + λ_lift  × max(0, 抬升)          ← 抬起奖励

λ_track >> λ_lift  ← 跟踪 MANO 是最高优先级!
```

**关键**: `λ_track` 设为最大权重，确保 RL 不会偏离 MANO 参考。之前 V1 是 λ_lift 太大（100），导致 RL 为了抬瓶子放弃了姿态跟踪。

### Step 4: 评估 + 渲染

- 回放 RL 策略
- 双栏对比: HaMeR 原视频 | MuJoCo RL 仿真
- 指标: lift_cm, residual_norm, MANO_tracking_error, contacts

---

## 与 RoboWheel 论文对比

| 步骤 | RoboWheel | V2 |
|------|-----------|-----|
| 手部重建 | HaMeR → MANO | **相同** |
| 参考轨迹 | MANO 全部 (pos+orient+fingers) | **相同** |
| 物体追踪 | FPose 6DoF | thumb/index 中点估计 |
| 穿透消除 | TSDF SDF 优化 | MuJoCo 接触物理 |
| RL 框架 | IsaacGym + ManipTrans PPO | **MuJoCo + SB3 PPO** |
| RL 输入 | 完整 MANO 参考 + 小残差 | **相同** |
| 奖励 | track + smooth + contact | **相同** |
| 并行 | 4096 envs (GPU) | 16 envs (CPU) |

---

## 预期改善

| 指标 | V1 | V2 预期 |
|------|-----|---------|
| 手的朝向 | 硬编码/不匹配视频 | **跟随 MANO** |
| 手的位置 | 手写 approach | **跟随 MANO wrist** |
| 手指弯曲 | 部分来自 MANO | **完全来自 MANO** |
| 视觉相似度 | 低 | **高** |
| 抬升 | 11.6cm | 可能降低（跟踪优先于抬升）|

---

*计划日期: 2026-04-02*
