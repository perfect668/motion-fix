"""Open GMR robot MJCF models in the native MuJoCo viewer.

Purpose:
    Inspect robot zero pose, joint axes, joint state, and joint ranges using
    MuJoCo's built-in viewer UI. This is a kinematic inspector, not a controller
    debugger: the native Control panel sliders are copied directly into hinge
    and slide joint qpos values, and the script calls mj_forward to update the
    rendered pose. It never uses mj_step or actuator dynamics in the viewer loop.

Typical usage:
    conda run --no-capture-output -n gmr python scripts/mujoco_joint_inspector.py \
        --robot ne01

    conda run --no-capture-output -n gmr python scripts/mujoco_joint_inspector.py \
        --robot unitree_g1

Headless checks:
    conda run --no-capture-output -n gmr python scripts/mujoco_joint_inspector.py \
        --robot ne01 --print-joints

    conda run --no-capture-output -n gmr python scripts/mujoco_joint_inspector.py \
        --check-all-robots

Viewer notes:
    Keep the MuJoCo left and right UI panels open. Use the built-in Control
    panel to set joint positions and the built-in joint/state panels to inspect
    qpos, qvel, axes, and limits. A temporary hidden XML is generated beside the
    source XML by default so relative mesh paths keep working; it replaces the
    model actuators with simple position sliders for every hinge/slide joint.
    The source XML is not modified, and the temporary XML is removed on exit
    unless --keep-generated-xml is set.

    By default the free-joint root is locked and lifted by --root-height-offset,
    so the robot stays suspended for kinematic inspection.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from general_motion_retargeting.params import (
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)


JOINT_TYPE_NAMES = {
    int(mj.mjtJoint.mjJNT_FREE): "free",
    int(mj.mjtJoint.mjJNT_BALL): "ball",
    int(mj.mjtJoint.mjJNT_SLIDE): "slide",
    int(mj.mjtJoint.mjJNT_HINGE): "hinge",
}


@dataclass(frozen=True)
class JointInfo:
    index: int
    name: str
    joint_type: int
    joint_type_name: str
    qpos_addr: int
    dof_addr: int
    body_id: int
    axis: np.ndarray
    limited: bool
    lower: float
    upper: float

    @property
    def is_hinge(self) -> bool:
        return self.joint_type == int(mj.mjtJoint.mjJNT_HINGE)

    @property
    def range_text(self) -> str:
        suffix = "" if self.limited else " (unlimited)"
        if self.is_hinge:
            return (
                f"{self.lower:+.4f}..{self.upper:+.4f} rad "
                f"({math.degrees(self.lower):+.1f}..{math.degrees(self.upper):+.1f} deg)"
                f"{suffix}"
            )
        return f"{self.lower:+.4f}..{self.upper:+.4f}{suffix}"


@dataclass(frozen=True)
class ActuatedJoint:
    actuator_id: int
    joint_id: int
    qpos_addr: int
    dof_addr: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a GMR robot model in MuJoCo's native kinematic viewer."
    )
    parser.add_argument(
        "--robot",
        default="unitree_g1_24dof",
        choices=sorted(ROBOT_XML_DICT.keys()),
        help="Robot key from general_motion_retargeting.params.ROBOT_XML_DICT.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Load this MJCF XML instead of a ROBOT_XML_DICT entry.",
    )
    parser.add_argument(
        "--list-robots",
        action="store_true",
        help="Print available ROBOT_XML_DICT keys and exit.",
    )
    parser.add_argument(
        "--print-joints",
        action="store_true",
        help="Print hinge/slide joints and exit without opening the viewer.",
    )
    parser.add_argument(
        "--check-all-robots",
        action="store_true",
        help="Load every ROBOT_XML_DICT XML and print nq/nv/joint counts.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated kinematic XML. Defaults to the source XML "
            "directory so relative mesh paths keep working. Custom directories "
            "may break relative mesh paths in some MJCF files."
        ),
    )
    parser.add_argument(
        "--unlimited-hinge-range",
        type=float,
        default=math.pi,
        help="Control half-range in radians for unlimited hinge joints.",
    )
    parser.add_argument(
        "--unlimited-slide-range",
        type=float,
        default=0.5,
        help="Control half-range for unlimited slide joints.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Enable MuJoCo transparent-robot visualization flag.",
    )
    parser.add_argument(
        "--free-camera",
        action="store_true",
        help="Do not initialize the camera from ROBOT_BASE_DICT.",
    )
    parser.add_argument(
        "--lock-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep free-joint root qpos/qvel fixed. Enabled by default.",
    )
    parser.add_argument(
        "--root-height-offset",
        type=float,
        default=0.35,
        help="Extra height added to free-joint root qpos z when --lock-root is enabled.",
    )
    parser.add_argument(
        "--keep-generated-xml",
        action="store_true",
        help="Do not delete the generated kinematic XML after the viewer exits.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Viewer sync frequency.",
    )
    return parser.parse_args()


def resolve_xml(args: argparse.Namespace) -> tuple[str, Path]:
    if args.xml is not None:
        return args.xml.stem, args.xml.expanduser().resolve()
    return args.robot, ROBOT_XML_DICT[args.robot].resolve()


def get_joint_name(model: mj.MjModel, joint_id: int) -> str:
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
    return name if name else f"joint_{joint_id}"


def get_body_name(model: mj.MjModel, body_id: int) -> str:
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


def is_inspectable_joint(model: mj.MjModel, joint_id: int) -> bool:
    joint_type = int(model.jnt_type[joint_id])
    return joint_type in {
        int(mj.mjtJoint.mjJNT_HINGE),
        int(mj.mjtJoint.mjJNT_SLIDE),
    }


def collect_joint_infos(model: mj.MjModel) -> list[JointInfo]:
    infos: list[JointInfo] = []
    for joint_id in range(model.njnt):
        if not is_inspectable_joint(model, joint_id):
            continue

        joint_type = int(model.jnt_type[joint_id])
        limited = bool(model.jnt_limited[joint_id])
        if limited:
            lower, upper = map(float, model.jnt_range[joint_id])
        else:
            lower, upper = 0.0, 0.0

        infos.append(
            JointInfo(
                index=joint_id,
                name=get_joint_name(model, joint_id),
                joint_type=joint_type,
                joint_type_name=JOINT_TYPE_NAMES.get(joint_type, str(joint_type)),
                qpos_addr=int(model.jnt_qposadr[joint_id]),
                dof_addr=int(model.jnt_dofadr[joint_id]),
                body_id=int(model.jnt_bodyid[joint_id]),
                axis=np.asarray(model.jnt_axis[joint_id], dtype=float).copy(),
                limited=limited,
                lower=lower,
                upper=upper,
            )
        )
    return infos


def print_joint_table(model: mj.MjModel, infos: Iterable[JointInfo]) -> None:
    print(
        "idx  name                          type   qpos dof  body"
        "                          axis(local)          range"
    )
    print("-" * 120)
    for info in infos:
        body_name = get_body_name(model, info.body_id)
        axis = " ".join(f"{v:+.3f}" for v in info.axis)
        print(
            f"{info.index:>3}  {info.name:<29} {info.joint_type_name:<6} "
            f"{info.qpos_addr:>4} {info.dof_addr:>3}  {body_name:<29} "
            f"[{axis}]  {info.range_text}"
        )


def collect_actuated_hinge_slide_joints(model: mj.MjModel) -> list[ActuatedJoint]:
    actuated: list[ActuatedJoint] = []
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mj.mjtTrn.mjTRN_JOINT):
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or not is_inspectable_joint(model, joint_id):
            continue
        actuated.append(
            ActuatedJoint(
                actuator_id=actuator_id,
                joint_id=joint_id,
                qpos_addr=int(model.jnt_qposadr[joint_id]),
                dof_addr=int(model.jnt_dofadr[joint_id]),
            )
        )
    return actuated


def clamp_controls_to_range(model: mj.MjModel, data: mj.MjData) -> None:
    if model.nu == 0:
        return
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    if not np.any(limited):
        return
    data.ctrl[limited] = np.clip(
        data.ctrl[limited],
        model.actuator_ctrlrange[limited, 0],
        model.actuator_ctrlrange[limited, 1],
    )


def set_controls_to_current_qpos(
    model: mj.MjModel,
    data: mj.MjData,
    actuated_joints: Iterable[ActuatedJoint],
) -> None:
    data.ctrl[:] = 0.0
    for actuated in actuated_joints:
        data.ctrl[actuated.actuator_id] = data.qpos[actuated.qpos_addr]
    clamp_controls_to_range(model, data)


def apply_control_sliders_to_qpos(
    model: mj.MjModel,
    data: mj.MjData,
    actuated_joints: Iterable[ActuatedJoint],
) -> None:
    clamp_controls_to_range(model, data)
    data.qvel[:] = 0.0
    for actuated in actuated_joints:
        data.qpos[actuated.qpos_addr] = data.ctrl[actuated.actuator_id]


def check_all_robots() -> int:
    failed = 0
    for robot, xml_path in sorted(ROBOT_XML_DICT.items()):
        try:
            model = mj.MjModel.from_xml_path(str(xml_path))
            infos = collect_joint_infos(model)
            print(
                f"[OK] {robot:<28} nq={model.nq:<3} nv={model.nv:<3} "
                f"njnt={model.njnt:<3} hinge_slide={len(infos):<3} "
                f"nu={model.nu:<3} xml={xml_path}"
            )
        except Exception as exc:  # pragma: no cover - asset dependent.
            failed += 1
            print(f"[FAIL] {robot:<28} {type(exc).__name__}: {exc}")
    return 1 if failed else 0


def freejoint_addresses(model: mj.MjModel) -> list[tuple[int, int]]:
    addresses = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mj.mjtJoint.mjJNT_FREE):
            continue
        addresses.append(
            (int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))
        )
    return addresses


def build_locked_root_qpos(
    model: mj.MjModel,
    root_height_offset: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    qpos = np.asarray(model.qpos0, dtype=float).copy()
    addresses = freejoint_addresses(model)
    for qadr, _dadr in addresses:
        qpos[qadr + 2] += root_height_offset
        qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    return qpos, addresses


def apply_locked_root(
    data: mj.MjData,
    locked_root_qpos: np.ndarray,
    root_addresses: list[tuple[int, int]],
) -> None:
    for qadr, dadr in root_addresses:
        data.qpos[qadr : qadr + 7] = locked_root_qpos[qadr : qadr + 7]
        data.qvel[dadr : dadr + 6] = 0.0


def set_camera(viewer, model: mj.MjModel, data: mj.MjData, robot_key: str) -> None:
    base_name = ROBOT_BASE_DICT.get(robot_key)
    body_id = -1
    if base_name is not None:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name)
    if body_id < 0:
        body_id = 1 if model.nbody > 1 else 0

    viewer.cam.lookat = data.xpos[body_id]
    viewer.cam.distance = VIEWER_CAM_DISTANCE_DICT.get(robot_key, 2.0)
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 135


def actuator_ctrl_range(
    info: JointInfo,
    unlimited_hinge_range: float,
    unlimited_slide_range: float,
) -> tuple[float, float]:
    if info.limited:
        return info.lower, info.upper
    if info.is_hinge:
        return -unlimited_hinge_range, unlimited_hinge_range
    return -unlimited_slide_range, unlimited_slide_range


def replace_actuators_with_kinematic_sliders(
    expanded_xml: Path,
    output_xml: Path,
    joint_infos: Iterable[JointInfo],
    unlimited_hinge_range: float,
    unlimited_slide_range: float,
) -> int:
    tree = ET.parse(expanded_xml)
    root = tree.getroot()
    for old_actuator in list(root.findall("actuator")):
        root.remove(old_actuator)
    actuator = ET.SubElement(root, "actuator")

    added = 0
    for info in joint_infos:
        lower, upper = actuator_ctrl_range(
            info,
            unlimited_hinge_range=unlimited_hinge_range,
            unlimited_slide_range=unlimited_slide_range,
        )
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"qpos_{info.name}",
                "joint": info.name,
                "kp": "1",
                "ctrlrange": f"{lower:.8g} {upper:.8g}",
                "ctrllimited": "true",
            },
        )
        added += 1

    ET.indent(tree, space="  ")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return added


def prepare_viewer_xml(
    xml_path: Path,
    args: argparse.Namespace,
) -> Path:
    source_model = mj.MjModel.from_xml_path(str(xml_path))
    joint_infos = collect_joint_infos(source_model)
    if args.tmp_dir is None:
        tmp_dir = xml_path.parent
        prefix = f".{xml_path.stem}"
    else:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = args.tmp_dir
        prefix = xml_path.stem

    output_xml = tmp_dir / f"{prefix}_kinematic_inspector.xml"
    expanded_xml = tmp_dir / f"{prefix}_expanded.xml"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mj.mj_saveLastXML(str(expanded_xml), source_model)
    added = replace_actuators_with_kinematic_sliders(
        expanded_xml=expanded_xml,
        output_xml=output_xml,
        joint_infos=joint_infos,
        unlimited_hinge_range=args.unlimited_hinge_range,
        unlimited_slide_range=args.unlimited_slide_range,
    )
    print(f"Generated kinematic inspector XML: {output_xml}")
    print(f"Added qpos sliders: {added}")
    if args.keep_generated_xml:
        print("Keeping generated kinematic inspector XML after exit.")
    return output_xml


def cleanup_generated_xml(viewer_xml: Path) -> None:
    try:
        viewer_xml.unlink(missing_ok=True)
        expanded_xml = viewer_xml.with_name(
            viewer_xml.name.replace("_kinematic_inspector.xml", "_expanded.xml")
        )
        expanded_xml.unlink(missing_ok=True)
    except OSError as exc:
        print(f"Warning: failed to remove generated XML: {exc}", file=sys.stderr)


def launch_native_viewer(
    robot_key: str,
    xml_path: Path,
    args: argparse.Namespace,
) -> None:
    viewer_xml = prepare_viewer_xml(xml_path, args)
    model = mj.MjModel.from_xml_path(str(viewer_xml))
    model.opt.gravity[:] = 0.0
    data = mj.MjData(model)
    actuated_joints = collect_actuated_hinge_slide_joints(model)

    if args.lock_root:
        qpos0, root_addresses = build_locked_root_qpos(
            model,
            root_height_offset=args.root_height_offset,
        )
    else:
        qpos0 = np.asarray(model.qpos0, dtype=float).copy()
        root_addresses = []

    data.qpos[:] = qpos0
    data.qvel[:] = 0.0
    if model.nu:
        set_controls_to_current_qpos(model, data, actuated_joints)
    if root_addresses:
        apply_locked_root(data, qpos0, root_addresses)
    mj.mj_forward(model, data)

    print(f"Opening MuJoCo kinematic viewer: robot={robot_key} xml={viewer_xml}")
    print(f"nq={model.nq} nv={model.nv} njnt={model.njnt} nu={model.nu}")
    print("Control sliders are written directly to qpos; mj_step is not used.")
    if args.lock_root and root_addresses:
        print(
            "Root lock enabled: free-joint root is fixed and lifted by "
            f"{args.root_height_offset:g} m."
        )
    print("Use MuJoCo's built-in UI panels for controls and joint/state inspection.")

    viewer = mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=True,
    )
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = True
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = args.transparent
    if not args.free_camera:
        set_camera(viewer, model, data, robot_key)

    dt = 1.0 / max(args.fps, 1.0)
    try:
        while viewer.is_running():
            start = time.monotonic()
            apply_control_sliders_to_qpos(model, data, actuated_joints)
            if root_addresses:
                apply_locked_root(data, qpos0, root_addresses)
            mj.mj_forward(model, data)
            viewer.sync()
            sleep_time = dt - (time.monotonic() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        viewer.close()
        if not args.keep_generated_xml:
            cleanup_generated_xml(viewer_xml)


def main() -> int:
    args = parse_args()
    if args.list_robots:
        for robot in sorted(ROBOT_XML_DICT.keys()):
            print(robot)
        return 0
    if args.check_all_robots:
        return check_all_robots()

    robot_key, xml_path = resolve_xml(args)
    if not xml_path.exists():
        print(f"XML not found: {xml_path}", file=sys.stderr)
        return 1

    model = mj.MjModel.from_xml_path(str(xml_path))
    infos = collect_joint_infos(model)
    if args.print_joints:
        print(f"robot={robot_key} xml={xml_path}")
        print(f"nq={model.nq} nv={model.nv} njnt={model.njnt} nu={model.nu}")
        print_joint_table(model, infos)
        return 0

    launch_native_viewer(robot_key, xml_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
