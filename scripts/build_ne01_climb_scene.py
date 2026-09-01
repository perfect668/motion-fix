"""Build a standalone desktop-NE01 MuJoCo scene with HoloSoMo climb boxes."""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-xml", type=pathlib.Path, required=True)
    parser.add_argument("--box-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--box-scale", type=float, default=0.7415730337078652)
    parser.add_argument("--box-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    args = parser.parse_args()

    tree = ET.parse(args.robot_xml)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("Robot XML must contain asset and worldbody")
    colors = ("0.3 0.7 0.9 0.65", "0.7 0.3 0.9 0.65", "0.9 0.7 0.3 0.65")
    scale = " ".join([str(args.box_scale)] * 3)
    offset = " ".join(map(str, args.box_offset))
    for i, color in enumerate(colors, start=1):
        mesh_path = (args.box_dir / "box_models" / f"box{i}.obj").resolve()
        ET.SubElement(asset, "mesh", name=f"climb_box_{i}", file=str(mesh_path), scale=scale)
        ET.SubElement(asset, "material", name=f"climb_box_mat_{i}", rgba=color)
        body = ET.SubElement(worldbody, "body", name=f"climb_box_{i}_body", pos=offset)
        ET.SubElement(
            body, "geom", name=f"climb_box_{i}_geom", type="mesh",
            mesh=f"climb_box_{i}", material=f"climb_box_mat_{i}",
            contype="1", conaffinity="1",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
