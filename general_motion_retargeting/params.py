"""NE01-only asset and viewer parameters."""

from pathlib import Path

HERE = Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = (HERE / ".." / "assets").resolve()
NE01_XML = ASSET_ROOT / "ne01" / "ne01.xml"

ROBOT_XML_DICT = {"ne01": NE01_XML}
ROBOT_BASE_DICT = {"ne01": "base_link"}
VIEWER_CAM_DISTANCE_DICT = {"ne01": 2.0}
