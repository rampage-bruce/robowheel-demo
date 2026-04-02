# RoboWheel 开源替代管线 Pipeline 架构

> 从互联网视频到双手灵巧手仿真抓取的完整技术管线
> 日期：2026-04-02

---

## Pipeline 总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RoboWheel 开源替代管线 Pipeline                       │
│                    (互联网视频 → 灵巧手仿真抓取)                         │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  互联网视频    │  pick_bottle.mp4 (15s, 640×360, 30fps)
  │  人双手抓水瓶  │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐
  │   MediaPipe   │────▶│    HaMeR     │  GPU ~2.6GB
  │  手部检测     │     │  手部3D重建   │  ViT-H backbone
  │  (CPU, 轻量)  │     │              │
  └──────────────┘     └──────┬───────┘
                              │
                    输出: MANO 参数 (151帧右手, 110帧左手)
                    - hand_pose: 15×(3×3) 旋转矩阵
                    - betas: 10D 形状
                    - global_orient: 3×3 全局朝向
                    - cam_t: 3D 相机位移
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐   ┌──────────────┐    ┌──────────────┐
  │   smplx       │   │ Hunyuan3D-2  │    │  nvblox      │
  │  关节重建      │   │ 物体3D重建    │    │ TSDF构建     │
  │              │   │              │    │              │
  │ MANO→16关节   │   │ 视频截帧→     │    │ 瓶子mesh→    │
  │ 左右手分离     │   │ AI生成mesh    │    │ 48视角深度→   │
  │              │   │ 2.5K顶点      │    │ GPU体素场     │
  └──────┬───────┘   └──────┬───────┘    └──────┬───────┘
         │                  │                    │
         │           (可选,当前用圆柱体)           │
         │                                       │
         ▼                                       │
  ┌──────────────────────────────────┐           │
  │    MuJoCo 场景构建 (MjSpec API)    │           │
  │                                  │           │
  │  ┌────────┐  ┌────────┐  ┌────┐ │           │
  │  │右Allegro│  │左Allegro│  │瓶子│ │           │
  │  │16 DOF  │  │16 DOF  │  │圆柱│ │           │
  │  │+6D base│  │+6D base│  │静态│ │           │
  │  └────────┘  └────────┘  └────┘ │           │
  │          44 actuators            │           │
  │          + 桌面 + 光照            │           │
  └──────────────┬───────────────────┘           │
                 │                               │
                 ▼                               │
  ┌──────────────────────────────────┐           │
  │    5阶段轨迹生成                    │           │
  │                                  │           │
  │  REACH (40帧)                    │           │
  │    手从远处→瓶子旁 (cubic ease)    │           │
  │    手指全开                       │           │
  │         ↓                        │           │
  │  APPROACH (0-25%)                │           │
  │    X轴接近瓶子 (-3cm)             │           │
  │    手指保持开                     │           │
  │         ↓                        │           │
  │  CLOSE (25-50%)                  │           │
  │    继续靠近 (-5cm)                │           │
  │    手指curl 0→0.6 rad            │           │
  │         ↓                        │           │
  │  GRASP (50-70%)                  │           │
  │    紧贴瓶身                       │           │
  │    手指curl 0.6→0.9 rad          │           │
  │         ↓                        │           │
  │  LIFT (70-100%)                  │           │
  │    Z+4cm 提起                    │           │
  │    手指保持0.9 rad               │           │
  └──────────────┬───────────────────┘           │
                 │                               │
                 ▼                               │
  ┌──────────────────────────────────┐           │
  │    nvblox GPU SDF 穿透修正        │◀──────────┘
  │                                  │
  │  逐帧逐指:                        │
  │    1. 查询指尖→瓶子 SDF 距离      │  GPU批量查询
  │    2. SDF < -0.5mm → 穿透!       │
  │    3. Binary search (6次迭代):    │
  │       找到最大安全curl值           │  精度 1/64
  │       不穿透的最大弯曲角度         │
  │    4. 设置关节角 = 安全值          │
  │                                  │
  │  结果: 46穿透 → 184关节修正       │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │    MuJoCo 渲染                    │
  │                                  │
  │  相机: 3/4视角 (az=145° el=-30°) │
  │  帧率: 10fps                     │
  │  分辨率: 640×480                  │
  │                                  │
  │  输出:                            │
  │  ├── bimanual_grasp.mp4          │
  │  ├── comparison.mp4 (双栏对比)    │
  │  ├── comparison.gif              │
  │  └── kf_*.jpg (关键帧)           │
  └──────────────────────────────────┘
```

---

## 各步骤详解

### 1. 手部 3D 重建 (HaMeR + MediaPipe)

| 项目 | 详情 |
|------|------|
| 输入 | 任意视频帧 |
| 检测 | MediaPipe Hand Landmarker (CPU) → 手部 2D bbox + 左右手分类 |
| 重建 | HaMeR (ViT-H, ~2.6GB GPU) → MANO 参数 |
| 输出 | 每帧 15 关节旋转矩阵 + 10D 形状 + 全局朝向 + 相机位移 |
| 性能 | 151 帧右手 + 110 帧左手 = 261 手部重建 |
| 脚本 | `hamer/run_hamer_mp.py` |

### 2. MANO 关节重建 (smplx)

| 项目 | 详情 |
|------|------|
| 输入 | MANO hand_pose (15×3×3) + betas (10D) + global_orient (3×3) |
| 方法 | smplx MANO 前向运动学 → 16 个 3D 关节位置 |
| 坐标变换 | MANO 相机空间 → 世界空间 (Y/Z 交换 + Z 偏移) |
| 双手 | 左手 X 轴翻转 (MANO 右手模型镜像) |
| 脚本 | `step_bimanual_grasp.py::load_mano_data()` |

### 3. 物体 3D 重建 (Hunyuan3D-2, 可选)

| 项目 | 详情 |
|------|------|
| 输入 | 视频截帧中的物体图片 |
| 模型 | tencent/Hunyuan3D-2 (DiT Flow Matching, ~16GB GPU) |
| 输出 | 完整 3D 纹理 mesh (172K 顶点 → 简化为 2.5K 顶点) |
| 尺度 | 根据已知物体尺寸缩放到真实大小 |
| 当前状态 | 可选（默认使用正确比例的圆柱体 r=3.3cm, h=20cm） |
| 脚本 | 独立推理 (Hunyuan3D-2/minimal_demo.py) |

### 4. nvblox GPU TSDF 构建

| 项目 | 详情 |
|------|------|
| 输入 | 瓶子 mesh |
| 方法 | 48 视角 (16 方位角 × 3 仰角) 光线追踪深度渲染 → nvblox TSDF 体素场 |
| 体素大小 | 3mm |
| 查询 | `mapper.query_layer(QueryType.TSDF, points_gpu)` → (距离, 权重) |
| 性能 | GPU 批量查询，比 trimesh CPU 快数量级 |
| 依赖 | nvblox_torch (从源码编译, CUDA 12.8, sm_86) |
| 脚本 | `step_bimanual_grasp.py::build_nvblox()` |

### 5. MuJoCo 场景构建

| 项目 | 详情 |
|------|------|
| API | MjSpec attach (动态组合 MJCF 模型) |
| 右手 | Allegro V3 right_hand.xml, 16 DOF + 6D base (3 slide + 3 hinge) |
| 左手 | Allegro V3 left_hand.xml, 同上 |
| 瓶子 | 静态圆柱体 (r=3.3cm, h=20cm, 蓝色, 无碰撞) |
| 桌面 | 静态平面 + 桌腿 |
| 朝向 | 右手: Z=π (手指朝左), 左手: 默认 (手指朝右), Y±0.25 掌心内倾 |
| 总计 | 44 actuators (16×2 手指 + 6×2 基座) |
| 脚本 | `step_bimanual_grasp.py::build_bimanual_scene()` |

### 6. 5 阶段轨迹生成

| 阶段 | 帧数 | 基座运动 | 手指 |
|------|------|---------|------|
| **REACH** | 0-39 (40帧) | cubic ease-in-out 从远处靠近 | 全开 (关节最小值) |
| **APPROACH** | 40-67 (25%) | X 轴 -3cm 向瓶子 | 保持 IK |
| **CLOSE** | 68-95 (50%) | X 轴 -5cm | curl 0→0.6 rad |
| **GRASP** | 96-122 (70%) | 紧贴 | curl 0.6→0.9 rad |
| **LIFT** | 123-149 (100%) | Z +4cm 抬起 | 保持 0.9 rad |

### 7. nvblox GPU SDF 穿透修正

| 项目 | 详情 |
|------|------|
| 检测阈值 | SDF < -0.5mm 且 weight > 0 |
| 修正方法 | **Binary search** (非 scipy 优化) |
| 搜索过程 | 对穿透指的关节: lo=安全值, hi=当前值, 6 次二分 → 1/64 精度 |
| 每次迭代 | 设 ctrl → MuJoCo step → 读指尖位置 → nvblox SDF 查询 |
| 结果 | 46 处穿透 → 184 个关节修正 |
| 优势 | 比 scipy L-BFGS-B 可靠 (梯度不可微也能工作) |
| 脚本 | `step_bimanual_grasp.py` 主循环 |

### 8. MuJoCo 渲染

| 项目 | 详情 |
|------|------|
| 渲染器 | MuJoCo Renderer (EGL offscreen) |
| 分辨率 | 640×480 |
| 帧率 | 10 fps |
| 相机 | 3/4 视角 (azimuth=145°, elevation=-30°, distance=0.42m) |
| 输出 | bimanual_grasp.mp4, comparison.mp4/gif, keyframes |
| 标注 | 阶段标签 ([REACH] [APPROACH] [CLOSE] [GRASP] [LIFT]) 叠加 |

---

## 工具栈

| 工具 | 来源 | 用途 | 安装方式 |
|------|------|------|---------|
| **HaMeR** | CVPR 2024 | 手部 MANO 3D 重建 | git clone + pip |
| **MediaPipe** | Google | 手部 2D 检测 | pip install |
| **smplx** | Max Planck | MANO 关节前向运动学 | pip install |
| **Hunyuan3D-2** | 腾讯 | 单图→3D mesh (可选) | git clone + pip |
| **nvblox** | NVIDIA | GPU TSDF 构建 + SDF 查询 | cmake 源码编译 |
| **nvblox_torch** | NVIDIA | nvblox PyTorch 绑定 | 同上 |
| **SPIDER** | Meta/FAIR | IK 运动学重定向 (示例验证) | pip install |
| **MuJoCo** | DeepMind | 物理仿真 + 渲染 | pip install |
| **Allegro Hand** | Wonik / mujoco_menagerie | 灵巧手模型 (16 DOF) | git clone |
| **trimesh** | 开源 | mesh 处理 + 光线追踪 | pip install |
| **scipy** | 开源 | 数值优化 | pip install |
| **ffmpeg** | 开源 | GIF/视频生成 | 系统安装 |

---

## 环境

| 项目 | 详情 |
|------|------|
| 服务器 | 192.168.77.25, 8× NVIDIA RTX A5000 (24GB) |
| Conda 环境 | `hawor` (HaMeR), `spider` (SPIDER+nvblox+渲染) |
| CUDA | 12.8 (系统), Driver 570.195.03 |
| Python | 3.10 (hawor), 3.12 (spider) |

---

## 与 RoboWheel 论文的对应关系

| 管线步骤 | RoboWheel 论文 | 本管线替代 | 差距 |
|----------|---------------|-----------|------|
| 手部重建 | HaMeR | HaMeR + MediaPipe | **相同** |
| 物体重建 | Hunyuan3D2 | 圆柱体 (Hunyuan3D2 可选) | 形状简化 |
| 物体追踪 | FPose 6DoF | 静态位置估计 | 无独立追踪 |
| 穿透消除 | TSDF SDF 优化 | **nvblox GPU TSDF + binary search** | 方法不同,效果接近 |
| RL 物理精化 | ManipTrans PPO | 5 阶段 staged trajectory | 无闭环力控 |
| 跨形态重定向 | 标准化动作空间 | MuJoCo MjSpec + staged curl | 无 MANO→Allegro IK |
| 仿真增强 | Isaac Sim 5维域随机化 | 无 | 缺失 |
| 渲染 | Isaac Sim 光追 | MuJoCo EGL | 质量差距 |

---

## 输出文件

```
/mnt/users/yjy/robowheel-demo/output/bimanual_grasp/
├── bimanual_grasp.mp4          # 仿真视频
├── comparison.mp4              # 双栏对比 (HaMeR | 仿真)
├── comparison.gif              # GIF 版
├── comparison_preview.jpg      # 预览帧
└── kf_*_{reach,approach,grasp,lift}.jpg  # 5阶段关键帧
```

---

## 脚本索引

| 脚本 | 功能 |
|------|------|
| `hamer/run_hamer_mp.py` | HaMeR + MediaPipe 手部重建 |
| `step_bimanual_grasp.py` | **主脚本**: 双手场景 + 轨迹 + nvblox SDF + 渲染 |
| `convert_mano_to_spider.py` | MANO → SPIDER 格式转换 |
| `step_sdf_nvblox.py` | nvblox 单手 SDF 精化 |
| `step_unified_grasp.py` | 单手 5 阶段 + nvblox |
| `step_approach_refine.py` | 侧面接近轨迹优化 |
| `step_sdf_refine.py` | trimesh CPU SDF 精化 (已被 nvblox 替代) |
| `step_allegro.py` | 单手 Allegro 仿真 |
| `step_multi_hand.py` | 多灵巧手对比 |
| `hoi_sim_demo.py` | HOI 3D 重建渲染 |
| `viz_3d.py` | Pyrender 3D 可视化 |

---

*文档日期：2026-04-02*
