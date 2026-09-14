# V5 Scene And Contact Pipeline

V5 formal scenes are static floors, box steps and ramps. JSON/OBJ/USD assets
are resolved once into prepared query/display/collision geometry; no complex
chair/bed/object interaction path is enabled in the terrain-only scope.
`MeshSceneField` performs triangle closest-point queries; `TerrainField`
handles floor and OBB primitives; `CompositeSceneField` unions static assets.

Only heel/toe channels are support evidence in V5. They use finite downward
support queries that reject side walls; contact timing uses distance, normal
speed and tangent speed with hysteresis. The solver adds soft normal/tangent
foot tasks and independent MuJoCo/terrain non-penetration inequalities for all
discovered robot proxies. The complete source sequence is admitted before any
debug frame truncation.
