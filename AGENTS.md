# NE01 WholeBody V5 Repository Guide

This branch contains the independent NE01 WholeBody V5 terrain-only pipeline. V5 is a new task-level orchestrator and does not call V3/V4 entry points. Historical V1–V4 files remain available as regression baselines and must not be removed or rewritten.

## Main Flow

Inputs are normalized by the V5 resolver/adapters into `CanonicalMotion`, admitted as complete flat or static stairs/ramp sequences, and solved by `WholeBodyRetargetSolver` using Mink/MuJoCo. V5 has foot-supported terrain contact only; complex object support (chairs/beds), hand-supported climbing and prone/kneeling actions are explicitly excluded. Outputs are 50 Hz NE01 PKL/NPZ motions with diagnostics and terrain metadata.

## Required Assets

Only `assets/ne01/` is version-controlled. `assets/body_models/` is intentionally ignored and must be supplied locally for SMPL-X/GRAIL conversion.

## Useful Commands

```bash
conda run --no-capture-output -n gmr python scripts/retarget.py --motion <input> --robot ne01 --output <output.pkl>
conda run --no-capture-output -n gmr python scripts/audit_v5_result.py <output.pkl> --replay
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Do not commit generated `work/`, `outputs/`, `runs/`, cache files or motion datasets. Use the repository's `.codex/instructions.md` for commit formatting.
