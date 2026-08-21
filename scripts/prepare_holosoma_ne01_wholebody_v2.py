"""Prepare an independent GMR-v2 asset/config backed by HoloSoMo's NE01 XML."""

from __future__ import annotations

import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
HOLO_XML = ROOT.parent / "holosoma" / "holosoma" / "src" / "holosoma" / "holosoma" / "data" / "robots" / "ne01_mujoco" / "mjcf" / "ne01.xml"
OUT_XML = ROOT / "assets" / "ne01" / "ne01_holosoma_wholebody_omni_gmr_v2.xml"
BASE_CONFIG = ROOT / "general_motion_retargeting" / "ik_configs" / "smplx_to_ne01_wholebody_omni_gmr_v2.json"
OUT_CONFIG = ROOT / "general_motion_retargeting" / "ik_configs" / "smplx_to_ne01_holosoma_wholebody_omni_gmr_v2.json"
BASE_IK_CONFIG = ROOT / "general_motion_retargeting" / "ik_configs" / "smplx_to_ne01.json"
OUT_IK_CONFIG = ROOT / "general_motion_retargeting" / "ik_configs" / "smplx_to_ne01_holosoma.json"


def add_sites_and_guards(xml: str) -> str:
    # HoloSoMo's XML has the same NE01 kinematic frames and meshes, but no
    # GMR-specific contact sites. Insert only the sites/geoms needed by v2.
    knee_sites = {
        "right": """
            <site name="ground_contact_right_knee" type="sphere" pos="-0.056 0 -0.282" size="0.008" rgba="1 0.5 0 1"/>
            <site name="ground_guard_right_knee_center" type="sphere" pos="-0.055 0 -0.303" size="0.008" rgba="1 0 0 1"/>
            <site name="ground_guard_right_knee_inner" type="sphere" pos="-0.060 0.028 -0.275" size="0.008" rgba="1 0 0 1"/>
            <site name="ground_guard_right_knee_outer" type="sphere" pos="-0.060 -0.028 -0.275" size="0.008" rgba="1 0 0 1"/>
            <site name="ground_guard_right_upper_shin" type="sphere" pos="-0.075 0 -0.230" size="0.008" rgba="1 0 0 1"/>
""",
        "left": """
            <site name="ground_contact_left_knee" type="sphere" pos="-0.055 0 -0.270" size="0.008" rgba="1 0.5 0 1"/>
            <site name="ground_guard_left_knee_center" type="sphere" pos="-0.055 0 -0.303" size="0.008" rgba="1 0 0 1"/>
            <site name="ground_guard_left_knee_inner" type="sphere" pos="-0.060 -0.028 -0.270" size="0.008" rgba="1 0 0 1"/>
            <site name="ground_guard_left_knee_outer" type="sphere" pos="-0.060 0.028 -0.270" size="0.008" rgba="1 0 0 1"/>
            <site name="ground_guard_left_upper_shin" type="sphere" pos="-0.075 0 -0.230" size="0.008" rgba="1 0 0 1"/>
""",
    }
    foot_sites = {
        "right": """
                  <body name="right_toe_link" pos="0.1 0 -0.02"/>
                  <geom name="right_foot_rear_left_collision" type="sphere" size="0.005" pos="-0.05 0.025 -0.035"/>
                  <geom name="right_foot_rear_right_collision" type="sphere" size="0.005" pos="-0.05 -0.025 -0.035"/>
                  <geom name="right_foot_front_left_collision" type="sphere" size="0.005" pos="0.12 0.03 -0.035"/>
                  <geom name="right_foot_front_right_collision" type="sphere" size="0.005" pos="0.12 -0.03 -0.035"/>
                  <site name="ground_contact_right_foot_center" type="sphere" pos="0.035 0 -0.039" size="0.009"/>
                  <site name="ground_guard_right_heel_inner" type="sphere" pos="-0.030 0.029 -0.039" size="0.006"/>
                  <site name="ground_guard_right_heel_outer" type="sphere" pos="-0.030 -0.029 -0.039" size="0.006"/>
                  <site name="ground_guard_right_forefoot_inner" type="sphere" pos="0.080 0.037 -0.039" size="0.006"/>
                  <site name="ground_guard_right_forefoot_outer" type="sphere" pos="0.080 -0.037 -0.039" size="0.006"/>
                  <site name="ground_guard_right_toe_inner" type="sphere" pos="0.130 0.030 -0.039" size="0.006"/>
                  <site name="ground_guard_right_toe_outer" type="sphere" pos="0.130 -0.030 -0.039" size="0.006"/>
""",
        "left": """
                  <body name="left_toe_link" pos="0.1 0 -0.02"/>
                  <geom name="left_foot_rear_left_collision" type="sphere" size="0.005" pos="-0.05 0.025 -0.035"/>
                  <geom name="left_foot_rear_right_collision" type="sphere" size="0.005" pos="-0.05 -0.025 -0.035"/>
                  <geom name="left_foot_front_left_collision" type="sphere" size="0.005" pos="0.12 0.03 -0.035"/>
                  <geom name="left_foot_front_right_collision" type="sphere" size="0.005" pos="0.12 -0.03 -0.035"/>
                  <site name="ground_contact_left_foot_center" type="sphere" pos="0.035 0 -0.039" size="0.009"/>
                  <site name="ground_guard_left_heel_inner" type="sphere" pos="-0.030 -0.029 -0.039" size="0.006"/>
                  <site name="ground_guard_left_heel_outer" type="sphere" pos="-0.030 0.029 -0.039" size="0.006"/>
                  <site name="ground_guard_left_forefoot_inner" type="sphere" pos="0.080 -0.037 -0.039" size="0.006"/>
                  <site name="ground_guard_left_forefoot_outer" type="sphere" pos="0.080 0.037 -0.039" size="0.006"/>
                  <site name="ground_guard_left_toe_inner" type="sphere" pos="0.130 -0.030 -0.039" size="0.006"/>
                  <site name="ground_guard_left_toe_outer" type="sphere" pos="0.130 0.030 -0.039" size="0.006"/>
""",
    }
    # HoloSoMo already names the relevant bodies using uppercase link names.
    replacements = [
        (r'<body name="HIP_YAW_R_LINK"([^>]*)>', r'<body name="HIP_YAW_R_LINK"\1>\n            <site name="ground_guard_right_lower_thigh" type="sphere" pos="-0.064 0 -0.240" size="0.008"/>'),
        (r'<body name="HIP_YAW_L_LINK"([^>]*)>', r'<body name="HIP_YAW_L_LINK"\1>\n            <site name="ground_guard_left_lower_thigh" type="sphere" pos="-0.063 0 -0.240" size="0.008"/>'),
        (r'<body name="KNEE_PITCH_R_LINK"([^>]*)>', r'<body name="KNEE_PITCH_R_LINK"\1>\n' + knee_sites["right"]),
        (r'<body name="KNEE_PITCH_L_LINK"([^>]*)>', r'<body name="KNEE_PITCH_L_LINK"\1>\n' + knee_sites["left"]),
        (r'<body name="ANKLE_ROLL_R_LINK"([^>]*)>', r'<body name="ANKLE_ROLL_R_LINK"\1>\n' + foot_sites["right"]),
        (r'<body name="ANKLE_ROLL_L_LINK"([^>]*)>', r'<body name="ANKLE_ROLL_L_LINK"\1>\n' + foot_sites["left"]),
    ]
    for pattern, replacement in replacements:
        xml, count = re.subn(pattern, replacement, xml, count=1)
        if count != 1:
            raise RuntimeError(f"Could not inject HoloSoMo NE01 asset marker: {pattern}")
    mesh_dir = HOLO_XML.parent.parent / "meshes" / "ne01"
    xml = xml.replace('meshdir="../meshes/ne01"', f'meshdir="{mesh_dir}"')
    return xml.replace('<mujoco model="NE01_scene">', '<mujoco model="NE01_holosoma_gmr_v2">')


def main() -> None:
    if not HOLO_XML.exists():
        raise FileNotFoundError(HOLO_XML)
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    # HoloSoMo uses uppercase BASE_LINK; all other configured body names are
    # already identical to its MuJoCo asset.
    for section in (config["ground_interaction_graph"]["regions"], config["ground_collision_points"]):
        for spec in section.values():
            if spec.get("robot_body") == "base_link":
                spec["robot_body"] = "BASE_LINK"
    config["_asset_note"] = "Independent HoloSoMo NE01 MuJoCo asset; generated from local holosoma data/robots/ne01_mujoco/mjcf/ne01.xml"
    ik_config = json.loads(BASE_IK_CONFIG.read_text(encoding="utf-8"))
    ik_config["robot_root_name"] = "BASE_LINK"
    ik_config["ik_priority_levels"] = [
        ["BASE_LINK" if name == "base_link" else name for name in level]
        for level in ik_config["ik_priority_levels"]
    ]
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = ik_config[table_name]
        if "base_link" in table:
            table["BASE_LINK"] = table.pop("base_link")
    OUT_XML.write_text(add_sites_and_guards(HOLO_XML.read_text(encoding="utf-8")), encoding="utf-8")
    OUT_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    OUT_IK_CONFIG.write_text(json.dumps(ik_config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_XML}")
    print(f"Wrote {OUT_CONFIG}")
    print(f"Wrote {OUT_IK_CONFIG}")


if __name__ == "__main__":
    main()
