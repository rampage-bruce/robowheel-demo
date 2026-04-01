# RoboWheel 简化版 Pipeline 执行记录

> 目标：从互联网视频出发，经 HOI 重建 → 灵巧手重定向 → MuJoCo 物理仿真，验证 RoboWheel 核心链路
> 日期：2026-03-31

---

## 总览

```
互联网视频 (pick_bottle, 15s)
    ↓
Step 1: HaMeR 手部 3D 重建 → MANO 参数 (151帧, 261只手)          ✅
    ↓
Step 2: HOI 3D 重建 (手 + 物体估计 + 场景渲染)                    ✅
    ↓
Step 3A: MANO → Franka 夹爪 (简化路线，已完成但不合理)             ✅
Step 3B: MANO → Shadow Hand 灵巧手 (正确路线)                     ✅
    ↓
Step 4: MuJoCo 物理仿真 + 渲染                                   ✅
    ↓
Step 5: 三栏对比 MP4                                             ✅
```

---

## Step 1: HaMeR 手部 3D 重建

**输入**: `pick_bottle` 视频 (15s, 640×360, 30fps)
**工具**: HaMeR (ViT-H backbone) + MediaPipe (手部检测)
**环境**: `hawor` conda env, GPU ~2.6GB

**输出**:
- `output/pick_bottle_video/mano_results.json` — 261 个手部 MANO 参数
  - `hand_pose`: 15 × (3×3) 旋转矩阵 (5指 × 3关节)
  - `betas`: 10D 手部形状
  - `global_orient`: (3×3) 全局朝向
  - `cam_t_full`: 3D 相机位移
- `output/pick_bottle_video/*_overlay.jpg` — 151 帧 HaMeR overlay

**脚本**: `hamer/run_hamer_mp.py`

---

## Step 2: HOI 3D 重建

**方法**: 从 MANO 手部位置推断物体（瓶子）位置，在 Pyrender 中渲染手+物体+桌面
**简化**: 物体用圆柱体近似（完整 RoboWheel 用 Hunyuan3D2 真实重建 + FPose 追踪）

**输出**: `output/hoi_demo/robowheel_hoi_demo.mp4` — 三栏对比

**脚本**: `hoi_sim_demo.py`

---

## Step 3A: MANO → Franka 夹爪 (已弃用)

**问题**: MANO 是 5 指 21 关节灵巧手，Franka 是 1-DoF 平行夹爪，信息丢失严重
**结论**: 只用了 thumb_tip + index_tip 两个点，丢弃了其他 3 根手指信息，不合理

**输出**: `output/franka_sim/franka_grasp.mp4` — Franka approach→grasp→lift (193步)
**脚本**: `step1_mano_to_grasp.py` + `step3_mujoco_sim.py`

---

## Step 3B: MANO → Shadow Hand 灵巧手 (正确路线)

**方法**: 建立 MANO 15 关节 → Shadow Hand 20 actuator 的映射

**关节映射关系**:

```
MANO (15 joints, 5 fingers × 3)          Shadow Hand (20 actuators)
─────────────────────────────────         ──────────────────────────
global_orient → wrist euler            →  rh_A_WRJ2 (deviation)
                                          rh_A_WRJ1 (flex)

thumb[0] CMC flex/spread              →  rh_A_THJ5 (rotation)
                                          rh_A_THJ4 (opposition)
                                          rh_A_THJ3 (twist, =0)
thumb[1] MCP flex                     →  rh_A_THJ2 (MCP flex)
thumb[2] IP flex                      →  rh_A_THJ1 (IP flex)

index[0] MCP flex/spread              →  rh_A_FFJ4 (spread)
                                          rh_A_FFJ3 (MCP flex)
index[1,2] PIP+DIP flex               →  rh_A_FFJ0 (coupled)

middle[0] MCP flex/spread             →  rh_A_MFJ4, rh_A_MFJ3
middle[1,2] PIP+DIP                   →  rh_A_MFJ0

ring[0] MCP flex/spread               →  rh_A_RFJ4, rh_A_RFJ3
ring[1,2] PIP+DIP                     →  rh_A_RFJ0

pinky[0] MCP flex/spread              →  rh_A_LFJ5 (metacarpal)
                                          rh_A_LFJ4 (spread)
                                          rh_A_LFJ3 (MCP flex)
pinky[1,2] PIP+DIP                    →  rh_A_LFJ0
```

**转换方式**:
1. MANO 每个关节的 3×3 旋转矩阵 → euler XYZ 分解
2. X 轴分量 = flexion/extension → Shadow 弯曲关节
3. Z 轴分量 = abduction/adduction → Shadow 展开关节
4. 时序平滑 (uniform_filter1d, window=5)
5. 角度 clip 到 Shadow Hand 关节限位

**输出**:
- `output/dexterous_sim/shadow_hand_grasp.mp4` — 斜视角
- `output/dexterous_sim/shadow_hand_top.mp4` — 俯视角
- `output/dexterous_sim/mano_to_shadow_combined.mp4` — 三栏对比
- `output/dexterous_sim/shadow_retarget.json` — 151帧 × 20 actuator 控制量

**脚本**: `step_dexterous_sim.py`

---

## Step 4-5: 最终输出文件

```
/mnt/users/yjy/robowheel-demo/output/
│
├── dexterous_sim/                          ← 灵巧手仿真（核心输出）
│   ├── mano_to_shadow_combined.mp4         # 三栏: HaMeR | Shadow侧视 | Shadow俯视
│   ├── shadow_hand_grasp.mp4               # Shadow Hand 仿真视频
│   ├── shadow_hand_top.mp4                 # 俯视角
│   ├── shadow_retarget.json                # 重定向数据 (151帧×20 actuators)
│   └── keyframe_*.jpg                      # 关键帧
│
├── franka_sim/                             ← Franka 夹爪仿真（对比参考）
│   ├── franka_grasp.mp4                    # Franka approach→grasp→lift
│   └── validation_result.json              # 物理验证结果
│
├── hoi_demo/                               ← HOI 3D 重建
│   ├── robowheel_hoi_demo.mp4              # 三栏: 原视频 | HaMeR | HOI 3D
│   └── hoi_3d_scene.mp4                    # 仅 3D 场景
│
├── pick_bottle_video/                      ← HaMeR 完整输出
│   ├── mano_results.json                   # 261 个手部 MANO 参数
│   ├── *_overlay.jpg                       # 151 帧 overlay
│   ├── hamer_overlay.mp4                   # overlay 视频
│   └── hamer_sidebyside.mp4               # 原图|overlay 对比
│
├── pick_bottle/                            ← 最初 9 帧测试（保留）
│   ├── mano_results.json                   # MANO 参数
│   ├── *_hand*.obj                         # 3D mesh 文件
│   └── *_overlay.jpg                       # overlay 图片
│
├── grasp_poses.json                        ← MANO→6DoF 抓取位姿
├── mano_to_franka_full_pipeline.mp4        ← Franka 三栏对比
├── hamer_3d_sim.mp4                        ← 纯手部 3D 渲染
└── hamer_combined.mp4                      ← 手部 overlay|3D 对比
```

---

## 与完整 RoboWheel 的差距

| 步骤 | 我们做的 | RoboWheel 完整版 |
|------|---------|-----------------|
| 手部重建 | HaMeR (相同) | HaMeR (相同) |
| 物体重建 | 圆柱体近似 | Hunyuan3D2 AI 生成真实 mesh |
| 物体追踪 | 从手位置推断 | FPose 独立 6DoF 追踪 |
| 穿透消除 | 无 | TSDF + SDF 优化 |
| 物理精化 | 无 | RL (ManipTrans) |
| 灵巧手重定向 | **MANO→Shadow Hand euler 映射** | 运动学相似性+接触保持约束 |
| 夹爪重定向 | MANO→Franka (thumb+index) | 标准化动作空间+CoTracker |
| 仿真增强 | 无 | Isaac Sim 5维域随机化 |

---

## 依赖

| 依赖 | 环境 | 状态 |
|------|------|------|
| HaMeR | `hawor` env + `/robowheel-demo/hamer/` | ✅ |
| MediaPipe | `hawor` env | ✅ |
| MuJoCo 3.6 | `v2r` env | ✅ |
| Shadow Hand MJCF | `mujoco_menagerie/shadow_hand/` | ✅ |
| Franka Panda MJCF | `mujoco_menagerie/franka_emika_panda/` | ✅ |
| Pyrender | `hawor` env | ✅ |
| scipy | `hawor`/`v2r` env | ✅ |
