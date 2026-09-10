# V5 Validation

`FinalValidator` replays every exported qpos through MuJoCo. It checks finite states, QP failures, all robot-scene geom pairs, all discovered terrain proxies, joint ranges and joint velocity limits. It also emits penetration percentiles, active collision counts, timing and per-channel contact ratios.

Example:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n gmr pytest -q
conda run --no-capture-output -n gmr python scripts/retarget.py \
  --motion /path/to/motion.pkl --robot ne01 --output work/v5_result.pkl
```

The formal output is written only when validation status is `VALID`. Use `--save-invalid-debug` to preserve a failed trajectory and its diagnostics for investigation.
