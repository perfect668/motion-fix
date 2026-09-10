"""Public V5 terrain limit contract.

The implementation is owned by the independent V5 solver module so its
MuJoCo indexing and active-set state cannot diverge from the solver.  This
facade keeps the documented import stable for tools and tests.
"""

from .wholebody_omni_gmr_v5 import TerrainNonPenetrationLimit

__all__ = ["TerrainNonPenetrationLimit"]
