"""Add HoloSoMo multi-box includes to the desktop NE01 MuJoCo XML."""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-xml", type=pathlib.Path, required=True)
    parser.add_argument("--box-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = args.box_dir / f"{args.robot_xml.stem}_w_multi_boxes.xml"
    tree = ET.parse(args.robot_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("NE01 XML must contain asset and worldbody")
    if compiler is not None:
        meshdir = pathlib.Path(compiler.get("meshdir", "meshes"))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((args.robot_xml.parent / meshdir).resolve()))
    ET.SubElement(asset, "include", file="box_assets.xml")
    ET.SubElement(worldbody, "include", file="box_body.xml")
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
