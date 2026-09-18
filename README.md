# NE01 WholeBody V4 — clean standalone edition

This directory is a self-contained cleanup of the validated WholeBody V4
pipeline.  It preserves the Omni interaction Laplacian, Mink inverse
kinematics, terrain contact and non-penetration, MuJoCo scene collision,
GRAIL USD/OBJ scene assembly, HoloSoMo input, SMPL-X/BVH/FBX adapters, exports,
and MuJoCo visualizers.

It deliberately contains no V3/V5 solver inheritance or entry-point monkey
patching.  `OmniSolverCore` owns shared numerical tasks and `WholeBodyV4`
extends it with V4 orientation, contact-temporal, and scene-collision logic.

## Local prerequisites

Use the existing `gmr` Conda environment.  `assets/body_models` is a symlink
to the locally supplied SMPL-X models in the original archive; keep that
directory available when converting SMPL-X or GRAIL data.

## Main commands

Flat or explicit box terrain input:

```bash
cd /home/user/桌面/ljy重定向_v4纯净版
conda run --no-capture-output -n gmr python scripts/retarget_motion.py \
  --motion <motion> --robot ne01 --save_path outputs/motion
```

GRAIL reconstruction with its object scene (USD/OBJ is found from metadata):

```bash
conda run --no-capture-output -n gmr python scripts/grail_retarget_scene.py \
  --motion <grail_recon.pkl> --save_path outputs/grail_scene --tgt_fps 50
```

HoloSoMo position sequence:

```bash
conda run --no-capture-output -n gmr python scripts/holosoma_retarget.py \
  --motion <motion.npz> --terrain <terrain.json> --save_path outputs/holosoma
```

Scene result viewer:

```bash
conda run --no-capture-output -n gmr python scripts/vis_robot_motion_scene.py \
  --xml outputs/.grail_scene_combined.xml --motion outputs/grail_scene.pkl --loop
```

## Layout

- `general_motion_retargeting/omni_solver_core.py`: interaction, terrain,
  collision, postural and Mink-QP core.
- `general_motion_retargeting/omni_solver.py`: standalone public V4 solver.
- `scripts/grail_retarget_scene.py`: GRAIL metadata → visual mesh → CoACD →
  combined MuJoCo scene → V4 solver.
- `scripts/retarget_motion.py`: automatic SMPL-X, HoloSoMo, BVH and FBX input
  detection.
- `general_motion_retargeting/ik_configs/ne01_v4.json`: V4-only configuration
  extending `v4_defaults.json` in this directory.

Generated scenes, outputs and CoACD caches are ignored by Git.
