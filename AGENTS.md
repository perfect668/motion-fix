# NE01 WholeBody V4 Repository Guide

This branch is the NE01-focused WholeBody V4 retargeting pipeline. Keep the V3 solver core as a dependency of V4; do not remove historical V1/V2/V3 files unless the V4 cleanup specification explicitly lists them.

## Main Flow

Inputs are normalized by `scripts/retarget_motion.py` into `CanonicalMotion`, then solved by `WholeBodyOmniGMRV4` using Mink/MuJoCo. `scripts/grail_to_robot_wholebody_v4.py` adds automatic USD/OBJ scene loading, surface sampling and CoACD collision geometry. Outputs are 50 Hz NE01 PKL/NPZ motions with diagnostics and scene metadata.

## Required Assets

Only `assets/ne01/` is version-controlled. `assets/body_models/` is intentionally ignored and must be supplied locally for SMPL-X/GRAIL conversion.

## Useful Commands

```bash
conda run --no-capture-output -n gmr python scripts/retarget_motion.py --motion <input> --robot ne01 --version v4 --save_path <output.pkl>
conda run --no-capture-output -n gmr python scripts/grail_to_robot_wholebody_v4.py --motion <grail.pkl> --save_path <output.pkl>
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Do not commit generated `work/`, `outputs/`, `runs/`, cache files or motion datasets. Use the repository's `.codex/instructions.md` for commit formatting.
