# V5 Coordinates And Scale

`SceneTransform(rotation, scale, translation)` is the only source-world to solver-world transform. Points receive `scale * rotation * p + translation`; normals receive rotation only and are renormalized. A paired scene is never resized by a human-height morphology ratio.

Motion-only morphology may scale body-relative target vectors, while scene-preserving and paired-scene jobs keep the source metric scene unchanged. Object pose trajectories are sampled on their explicit timestamps and stored in the output for replay.
