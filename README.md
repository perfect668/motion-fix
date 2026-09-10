# NE01 WholeBody V5 Retargeting

This repository contains an independent NE01 WholeBody V5 task pipeline. It accepts SMPL-X, BVH, FBX, HoloSoMo and GRAIL motions through a CanonicalMotion adapter registry, resolves scene ownership before solving, and combines interaction preservation, source-scene contacts and MuJoCo scene collision. V1-V4 entry points remain available as regression baselines.

## Prerequisites

- Python 3.10 and the `gmr` environment dependencies from `setup.py`.
- A local SMPL-X body model for SMPL-X and GRAIL inputs.
- Blender for binary FBX conversion.
- `usd-core`, `trimesh` and `coacd` for GRAIL/USD scene processing.

## Entry Points

The V5 unified task entry point is `scripts/retarget.py`:

```bash
conda run --no-capture-output -n gmr python scripts/retarget.py \
  --motion <motion> --robot ne01 --output work/v5_result.pkl
```

For an explicit scene relation, use a JSON job manifest so the resolver can
distinguish shared source scenes from source-to-target bindings. Complex GRAIL
USD/OBJ assets are loaded, sampled and decomposed automatically; no hand-made
collision XML is required.

The historical V4 entry point remains available for regression comparisons:

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

V5 output contains NE01 `qpos`, robot joint/DOF trajectories, final-FK body states, contact schedule, terrain metadata, the single scene transform and diagnostics. Formal PKL/NPZ files are written atomically only after final validation; failed runs are not silently exported. Generated outputs belong in `work/`, `outputs/` or `runs/`, which are ignored by Git.

More detail is in `docs/v5_architecture.md`, `docs/v5_data_contracts.md`,
`docs/v5_coordinate_and_scale_policy.md`, `docs/v5_scene_and_contact_pipeline.md`
and `docs/v5_validation.md`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
