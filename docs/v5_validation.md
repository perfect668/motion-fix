# V5 Validation

`FinalValidator` replays every exported qpos through MuJoCo. It checks finite states, QP failures, all robot-scene geom pairs, all discovered terrain proxies, joint ranges and joint velocity limits. It also emits penetration percentiles, active collision counts, timing and per-channel contact ratios.

Example:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n gmr pytest -q
conda run --no-capture-output -n gmr python scripts/retarget.py \
  --motion /path/to/motion.pkl --robot ne01 --output work/v5_result.pkl
```

Batch (each complete input is independent):

```bash
conda run --no-capture-output -n gmr python scripts/retarget_batch.py \
  --input walk_a.npz --input walk_b.bvh \
  --output-dir work/v5_batch --report work/v5_batch/report.json
```

The terrain viewer can replay a valid or explicit `INVALID_DEBUG` artifact
without applying a hidden root/floor correction:

```bash
conda run --no-capture-output -n gmr python scripts/vis_robot_motion_terrain.py \
  --motion work/v5_result.pkl
```

The formal output is written only when validation status is `VALID`. Use `--save-invalid-debug` to preserve a failed trajectory and its diagnostics for investigation.

For an independent replay (which ignores the stored status/checks and reloads the
bound model, terrain and qpos), run:

```bash
conda run --no-capture-output -n gmr python scripts/audit_v5_result.py \
  work/v5_result.pkl --replay --json work/v5_result.replay.json
```

Replay reports both geometric penetration (`minimum_signed_distance`) and
safety-margin violation (`minimum_terrain_slack`). They are intentionally
separate quantities. A `--max_frames` debug run never bypasses full-sequence
scope admission; excluded or unresolved inputs receive a `.status.json` report
and no formal motion output.
