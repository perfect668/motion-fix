# V5 Data Contracts

`CanonicalMotion` stores finite `[T,J,3]` positions in metres, right-handed `+Z` coordinates, strictly increasing timestamps, unique source names, optional world quaternions and an orientation-valid mask. Missing toe landmarks are not synthesized from a knee vector.

`ContactPlan` stores independent heel/toe channels, source anchors, finite
surface IDs, normals, state (`NONE`, `STATIC`, or `SLIDING`), confidence and
provenance. A contact episode is source-scene evidence; it is not inferred
from robot qpos. Butt/back/palm/knee support is outside V5 scope and is
reported as excluded rather than silently approximated.

`RetargetResult` contains robot qpos plus forward-kinematics-derived robot joint/dof names, body states, velocities, scene metadata, diagnostics and validation status. Invalid runs are written only as `.INVALID_DEBUG.*` when explicitly requested.
