"""Build a HoloSoMo-native NE01 asset with G1-equivalent proxy links."""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET


FOOT_PROXIES = {
    "ankle_roll_sphere_1_link": (-0.055, 0.030, -0.034),
    "ankle_roll_sphere_2_link": (-0.055, -0.030, -0.034),
    "ankle_roll_sphere_3_link": (0.110, 0.030, -0.034),
    "ankle_roll_sphere_4_link": (0.110, -0.030, -0.034),
    "ankle_roll_sphere_5_link": (0.140, 0.000, -0.034),
}


def _xyz(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def _find_xml_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"MuJoCo body not found: {name}")
    return body


def _append_xml_proxy(parent: ET.Element, name: str, pos: tuple[float, float, float]) -> None:
    if parent.find(f"body[@name='{name}']") is None:
        ET.SubElement(parent, "body", name=name, pos=_xyz(pos))


def build_xml(source: pathlib.Path, output: pathlib.Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = pathlib.Path(compiler.get("meshdir", "meshes"))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((source.parent / meshdir).resolve()))

    for side, suffix in (("left", "L"), ("right", "R")):
        foot = _find_xml_body(root, f"ANKLE_ROLL_{suffix}_LINK")
        for proxy_suffix, pos in FOOT_PROXIES.items():
            _append_xml_proxy(foot, f"{side}_{proxy_suffix}", pos)

    _append_xml_proxy(_find_xml_body(root, "HAND_YAW_L_LINK"), "left_sphere_hand_link", (0.0, -0.004, -0.080))
    _append_xml_proxy(_find_xml_body(root, "HAND_YAW_R_LINK"), "right_sphere_hand_link", (0.0, 0.004, -0.080))
    _append_xml_proxy(_find_xml_body(root, "base_link"), "pelvis_contour_link", (0.0, 0.0, 0.0))

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def _find_urdf_link(root: ET.Element, name: str) -> ET.Element:
    link = root.find(f"link[@name='{name}']")
    if link is None:
        raise RuntimeError(f"URDF link not found: {name}")
    return link


def _append_urdf_proxy(
    root: ET.Element,
    parent_name: str,
    name: str,
    pos: tuple[float, float, float],
) -> None:
    if root.find(f"link[@name='{name}']") is not None:
        return
    ET.SubElement(root, "link", name=name)
    joint = ET.SubElement(root, "joint", name=f"{name}_joint", type="fixed")
    ET.SubElement(joint, "origin", xyz=_xyz(pos), rpy="0 0 0")
    ET.SubElement(joint, "parent", link=parent_name)
    ET.SubElement(joint, "child", link=name)


def build_urdf(source: pathlib.Path, output: pathlib.Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = pathlib.Path(mesh.get("filename", ""))
        if filename and not filename.is_absolute():
            mesh.set("filename", str((source.parent / filename).resolve()))

    for side, suffix in (("left", "L"), ("right", "R")):
        parent = f"ANKLE_ROLL_{suffix}_LINK"
        _find_urdf_link(root, parent)
        for proxy_suffix, pos in FOOT_PROXIES.items():
            _append_urdf_proxy(root, parent, f"{side}_{proxy_suffix}", pos)

    _append_urdf_proxy(root, "HAND_YAW_L_LINK", "left_sphere_hand_link", (0.0, -0.004, -0.080))
    _append_urdf_proxy(root, "HAND_YAW_R_LINK", "right_sphere_hand_link", (0.0, 0.004, -0.080))
    _append_urdf_proxy(root, "base_link", "pelvis_contour_link", (0.0, 0.0, 0.0))

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    default_source = pathlib.Path.home() / "桌面" / "ne01-robot-assets"
    default_output = (
        root.parent
        / "holosoma"
        / "holosoma"
        / "src"
        / "holosoma_retargeting"
        / "holosoma_retargeting"
        / "models"
        / "ne01"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=pathlib.Path, default=default_source)
    parser.add_argument("--output-dir", type=pathlib.Path, default=default_output)
    args = parser.parse_args()

    output_urdf = args.output_dir / "ne01_24dof_holosoma.urdf"
    output_xml = args.output_dir / "ne01_24dof_holosoma.xml"
    build_urdf(args.source_dir / "main_g1_aligned_collision_v2_0810.urdf", output_urdf)
    build_xml(args.source_dir / "ne01_24dof_fk.xml", output_xml)
    print(f"Wrote {output_urdf}")
    print(f"Wrote {output_xml}")


if __name__ == "__main__":
    main()
