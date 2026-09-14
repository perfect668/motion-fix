# WholeBody V5 Architecture

V5 is an independent task-level pipeline. The flow is:

`RetargetJob -> BundleResolver -> MotionAdapter -> SceneLoader -> TransformPlan -> MorphologyMapper -> SourceContactDetector -> WholeBodyRetargetSolver -> FinalValidator -> AtomicExporter`.

The solver receives canonical semantic frames and a fixed per-frame contact plan. It never detects a dataset from a filename and never calls a V1-V4 retargeter. Legacy versions remain available as regression baselines.

`MOTION_ONLY` creates an analytic floor. `SHARED_SCENE` applies one transform to human and static terrain. V5 formally admits only foot-supported flat motion and static stairs/ramps; complex object support and dynamic scenes are rejected before solving. The source contact detector emits independent heel/toe episodes, while terrain non-penetration inspects the complete proxy set even when its adaptive active set is sparse. `SOURCE_TO_TARGET` and `TARGET_ONLY_WITH_EXPLICIT_BINDING` remain schema-compatible legacy relations but are not silently converted into a V5 terrain task.

The solver uses one effective configuration (defaults, config inheritance, job overrides, then explicit CLI policy). Terrain queries expose both signed surface distance and finite support-hit state. Root support correction is feet-only, bounded, and robust to conflicting stair facets; it is not a replacement for hard non-penetration. Final validation replays every exported qpos and measures all terrain proxies rather than trusting the last QP active set.

Runnable job examples are in `examples/v5/` (`flat_motion.job.json`,
`stairs_motion.job.json`, and `ramp_motion.job.json`). The chair job is an
explicit negative example and is rejected as a complete sequence before any
solver work. Batch execution is provided by `scripts/retarget_batch.py`; each
input receives its own status/report and one failure does not truncate or alter
another input.
