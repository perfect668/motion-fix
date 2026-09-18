from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_standalone_v4_has_no_historical_solver_or_entrypoint_imports():
    forbidden = (
        "wholebody_omni_gmr_v3",
        "WholeBodyOmniGMRV3",
        "WholeBodyOmniGMRV4",
        "RETARGETER_CLASS",
        "CONTACT_SURFACE_PROVIDER",
        "SCENE_CONTACT_POSTPROCESS",
    )
    sources = list((ROOT / "general_motion_retargeting").rglob("*.py"))
    sources.extend((ROOT / "scripts").glob("*.py"))
    text = "\n".join(path.read_text() for path in sources)
    assert not [token for token in forbidden if token in text]


def test_v4_configuration_extends_only_local_v4_defaults():
    config = (ROOT / "general_motion_retargeting/ik_configs/ne01_v4.json").read_text()
    assert '"extends": "v4_defaults.json"' in config
