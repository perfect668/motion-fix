"""Task bundle resolution for WholeBody V5."""

from .bundle_resolver import BundleResolver, GenericBundleResolver, GrailBundleResolver, HoloSoMoBundleResolver
from ..motion_adapters import detect_motion_format


def resolver_for_motion(motion, motion_format="auto"):
    kind = detect_motion_format(motion) if motion_format == "auto" else str(motion_format)
    if kind.startswith("grail"):
        return GrailBundleResolver()
    if kind.startswith("holosoma"):
        return HoloSoMoBundleResolver()
    return GenericBundleResolver()

__all__ = ["BundleResolver", "GenericBundleResolver", "GrailBundleResolver", "HoloSoMoBundleResolver", "resolver_for_motion"]
