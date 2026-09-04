# NE01 WholeBody V4 Retargeting

This repository contains the NE01 WholeBody V4 retargeting pipeline. It accepts SMPL-X, BVH, FBX, HoloSoMo and GRAIL motions. V4 inherits the V3 Mink solver core and adds format adapters, terrain-aware contacts and scene collision handling.

## Prerequisites

- Python 3.10 and the `gmr` environment dependencies from `setup.py`.
- A local SMPL-X body model for SMPL-X and GRAIL inputs.
- Blender for binary FBX conversion.
- `usd-core`, `trimesh` and `coacd` for GRAIL/USD scene processing.

## Entry Points

The unified motion entry point is `scripts/retarget_motion.py`:

```bash
conda run --no-capture-output -n gmr python scripts/retarget_motion.py \
  --motion <motion> --robot ne01 --version v4 --tgt_fps 50 \
  --save_path <output.pkl>
```

HoloSoMo terrain motions use `scripts/holosoma_to_robot_wholebody_v4.py`. GRAIL motions with complex objects use `scripts/grail_to_robot_wholebody_v4.py`:

```bash
conda run --no-capture-output -n gmr python scripts/grail_to_robot_wholebody_v4.py \
  --motion <grail_recon.pkl> --save_path <output.pkl> --tgt_fps 50
```

The output PKL contains NE01 `qpos`, root and DOF trajectories, contact schedule, terrain metadata, scene transform and diagnostics. Generated outputs belong in `work/`, `outputs/` or `runs/`, which are ignored by Git.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
