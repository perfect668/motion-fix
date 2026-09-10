# WholeBody V5 Architecture

V5 is an independent task-level pipeline. The flow is:

`RetargetJob -> BundleResolver -> MotionAdapter -> SceneLoader -> TransformPlan -> MorphologyMapper -> SourceContactDetector -> WholeBodyRetargetSolver -> FinalValidator -> AtomicExporter`.

The solver receives canonical semantic frames and a fixed per-frame contact plan. It never detects a dataset from a filename and never calls a V1-V4 retargeter. Legacy versions remain available as regression baselines.

`MOTION_ONLY` creates an analytic floor. `SHARED_SCENE` applies one transform to human and scene. `SOURCE_TO_TARGET` requires an explicit contact binding. `TARGET_ONLY_WITH_EXPLICIT_BINDING` requires explicit contact records because there is no source geometry from which to infer them.
