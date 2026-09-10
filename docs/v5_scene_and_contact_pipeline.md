# V5 Scene And Contact Pipeline

OBJ/USD assets are loaded into one triangle mesh for visual and query use. CoACD receives that same mesh and its cache manifest is used to build the combined MuJoCo model. `MeshSceneField` performs triangle closest-point queries; `TerrainField` handles floor and OBB primitives; `CompositeSceneField` unions multiple assets.

Foot channels use supportable upward surfaces, while generic body channels use nearest surfaces. Contact timing uses distance, normal speed and tangent speed with hysteresis. The solver adds soft normal/tangent contact tasks and independent MuJoCo robot-scene non-penetration inequalities.
