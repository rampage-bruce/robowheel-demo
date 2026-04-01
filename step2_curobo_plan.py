"""
Step 2: CuRobo trajectory planning for MANO-derived grasp pose.
Run inside curobo Docker: docker run --gpus all -v ... curobo:v0.7.4 python step2_curobo_plan.py
"""
import torch, json

from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.math import Pose
from curobo.types.robot import JointState

# Load grasp pose from Step 1
with open("/workspace/robowheel-demo/output/grasp_poses.json") as f:
    grasp_data = json.load(f)

targets_data = grasp_data["curobo_targets"]
obj_pos = targets_data["object_pos"]

print(f"Object at: {obj_pos}")
print(f"Grasp width: {grasp_data['franka_grasp_width_m']*100:.1f} cm")
print(f"Source: MANO frame {grasp_data['source_frame']}")

# Setup CuRobo
tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

# World: ground + table (with bottle as obstacle to avoid)
world = WorldConfig(cuboid=[
    Cuboid(name="ground", pose=[0, 0, -0.025, 1, 0, 0, 0], dims=[2, 2, 0.05]),
    Cuboid(name="table", pose=[0.4, 0.0, 0.25, 1, 0, 0, 0], dims=[0.6, 0.8, 0.5]),
])

config = MotionGenConfig.load_from_robot_config(
    "franka.yml", world, tensor_args, interpolation_dt=0.01)
mg = MotionGen(config)
mg.warmup(warmup_js_trajopt=False)
print("CuRobo ready!", flush=True)

home_js = tensor_args.to_device(
    torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]]))
plan_cfg = MotionGenPlanConfig(max_attempts=20, enable_graph=True, enable_opt=True)

# Three-phase trajectory from MANO-derived targets
phases = [
    ("approach", targets_data["approach"]["position"],
     targets_data["approach"]["quaternion_wxyz"], home_js),
    ("grasp", targets_data["grasp"]["position"],
     targets_data["grasp"]["quaternion_wxyz"], None),
    ("lift", targets_data["lift"]["position"],
     targets_data["lift"]["quaternion_wxyz"], None),
]

results = {}
last_js = home_js

for name, pos, q, start_override in phases:
    start = JointState.from_position(
        start_override if start_override is not None else last_js)
    goal = Pose(
        position=tensor_args.to_device(torch.tensor([pos])),
        quaternion=tensor_args.to_device(torch.tensor([q])),
    )
    result = mg.plan_single(start, goal, plan_cfg)
    ok = result.success.item()
    print(f"{name}: pos={[round(p,3) for p in pos]}, quat={[round(q_,3) for q_ in q]} -> {'OK' if ok else 'FAIL'}", flush=True)

    if ok:
        traj = result.get_interpolated_plan()
        results[name] = traj.position.cpu().numpy().tolist()
        last_js = traj.position[-1:]
        print(f"  {len(results[name])} steps", flush=True)
    else:
        print("  FAILED - trying with default downward orientation", flush=True)
        # Fallback: use standard downward quaternion
        fallback_q = [0.0, 1.0, 0.0, 0.0]
        goal_fb = Pose(
            position=tensor_args.to_device(torch.tensor([pos])),
            quaternion=tensor_args.to_device(torch.tensor([fallback_q])),
        )
        result_fb = mg.plan_single(start, goal_fb, plan_cfg)
        ok_fb = result_fb.success.item()
        print(f"  fallback: {ok_fb}", flush=True)
        if ok_fb:
            traj = result_fb.get_interpolated_plan()
            results[name] = traj.position.cpu().numpy().tolist()
            last_js = traj.position[-1:]
            print(f"  {len(results[name])} steps (fallback)", flush=True)
        else:
            print("  BOTH FAILED, stopping", flush=True)
            break

total = sum(len(v) for v in results.values())
print(f"\nTotal: {total} steps, phases: {list(results.keys())}", flush=True)

output = {
    "source": "MANO_HaMeR → CuRobo",
    "source_frame": grasp_data["source_frame"],
    "object_pos": obj_pos,
    "grasp_width": grasp_data["franka_grasp_width_m"],
    "phases": results,
    "total_steps": total,
}

out_path = "/workspace/robowheel-demo/output/franka_trajectory.json"
with open(out_path, "w") as f:
    json.dump(output, f)
print(f"Saved to {out_path}", flush=True)
