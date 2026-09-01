"""Visualize a terrain-aware NE01 trajectory and its contact diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import time

import imageio
import mujoco as mj
import mujoco.viewer
import numpy as np


STATE_COLORS = {
    "NONE": np.array([0.45, 0.45, 0.45, 0.65]),
    "STATIC": np.array([0.05, 0.9, 0.2, 1.0]),
    "SLIDING": np.array([1.0, 0.55, 0.05, 1.0]),
}


def _add_geom(scene, geom_type, size, pos, mat, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mj.mjv_initGeom(
        scene.geoms[scene.ngeom], geom_type, np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float), np.asarray(mat, dtype=float).reshape(9),
        np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def _add_arrow(scene, start, end, color) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mj.mjv_connector(
        scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_ARROW, 0.006,
        np.asarray(start, dtype=float), np.asarray(end, dtype=float),
    )
    scene.geoms[scene.ngeom].rgba[:] = color
    scene.ngeom += 1


def _draw_overlay(scene, terrain: dict, schedule: list, diagnostics: list, frame: int) -> None:
    for primitive in terrain.get("primitives", []):
        _add_geom(
            scene, mj.mjtGeom.mjGEOM_BOX, primitive["half_extents"], primitive["center"],
            primitive["rotation"], [0.25, 0.38, 0.58, 0.42],
        )
    if frame < len(schedule):
        for contact in schedule[frame].get("contacts", {}).values():
            state = str(contact.get("state", "NONE"))
            color = STATE_COLORS.get(state, STATE_COLORS["NONE"])
            human = np.asarray(contact["human_point_solver"], dtype=float)
            surface = np.asarray(contact["surface_point_solver"], dtype=float)
            normal = np.asarray(contact["surface_normal_solver"], dtype=float)
            _add_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, [0.018] * 3, human, np.eye(3), color)
            _add_arrow(scene, surface, surface + 0.09 * normal, color)
    if frame < len(diagnostics):
        active = set(diagnostics[frame].get("active_candidates", []))
        for name, item in diagnostics[frame].get("slacks", {}).items():
            point = np.asarray(item.get("point", [0, 0, 0]), dtype=float)
            if float(item.get("signed_distance", 0.0)) < 0.0:
                color, radius = [1.0, 0.0, 0.0, 1.0], 0.014
            elif name in active:
                color, radius = [1.0, 0.9, 0.05, 0.95], 0.010
            else:
                color, radius = [0.25, 0.8, 1.0, 0.35], 0.006
            _add_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, [radius] * 3, point, np.eye(3), color)


def _load_motion(path: Path):
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    if "qpos" in motion:
        qpos = np.asarray(motion["qpos"], dtype=float)
    else:
        qpos = np.concatenate((
            np.asarray(motion["root_pos"]),
            np.asarray(motion["root_rot"])[:, [3, 0, 1, 2]],
            np.asarray(motion["dof_pos"]),
        ), axis=1)
    return motion, qpos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    motion, qpos = _load_motion(args.motion)
    xml = args.xml or Path(motion["robot_xml"])
    model = mj.MjModel.from_xml_path(str(xml))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion nq={qpos.shape[1]}, model nq={model.nq}")
    data = mj.MjData(model)
    fps = float(motion["fps"])
    terrain = motion.get("terrain_primitives", {})
    schedule = motion.get("contact_schedule", [])
    diagnostics = motion.get("terrain_diagnostics", [])

    if args.video_path is not None:
        args.video_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mj.Renderer(model, width=args.width, height=args.height)
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.distance = 3.0
        camera.azimuth = 145.0
        camera.elevation = -18.0
        with imageio.get_writer(args.video_path, fps=fps, codec="libx264") as writer:
            for frame, pose in enumerate(qpos):
                data.qpos[:] = pose
                mj.mj_forward(model, data)
                camera.lookat[:] = data.xpos[model.body("base_link").id] + np.array([0.0, 0.0, 0.25])
                renderer.update_scene(data, camera=camera)
                _draw_overlay(renderer.scene, terrain, schedule, diagnostics, frame)
                writer.append_data(renderer.render())
        renderer.close()
        print(f"Saved terrain visualization video to {args.video_path}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -18.0
        frame = 0
        while viewer.is_running():
            started = time.monotonic()
            data.qpos[:] = qpos[frame]
            mj.mj_forward(model, data)
            viewer.cam.lookat[:] = data.xpos[model.body("base_link").id] + np.array([0.0, 0.0, 0.25])
            viewer.user_scn.ngeom = 0
            _draw_overlay(viewer.user_scn, terrain, schedule, diagnostics, frame)
            viewer.sync()
            frame += 1
            if frame >= len(qpos):
                if not args.loop:
                    break
                frame = 0
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
