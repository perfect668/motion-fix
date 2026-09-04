"""Soft terrain contact, sole orientation, and temporal support tasks."""

from __future__ import annotations

import mujoco as mj
import numpy as np
from mink.tasks.task import Task


def tangent_basis(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    reference = np.array([1.0, 0.0, 0.0]) if abs(normal[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    first = np.cross(normal, reference)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(normal, first)
    return np.column_stack((first, second))


class RobotPointGroup:
    def __init__(self, model: mj.MjModel, specification: dict) -> None:
        self.model = model
        self.site_ids = np.asarray([model.site(name).id for name in specification.get("sites", [])], dtype=int)
        self.body_id = None
        self.offsets = np.empty((0, 3))
        if "body" in specification:
            self.body_id = model.body(specification["body"]).id
            self.offsets = np.asarray(specification.get("offsets", [[0, 0, 0]]), dtype=float).reshape((-1, 3))
        if len(self.site_ids) == 0 and self.body_id is None:
            raise ValueError("RobotPointGroup requires sites or body offsets")

    def point(self, configuration) -> np.ndarray:
        points = []
        if len(self.site_ids):
            points.extend(configuration.data.site_xpos[self.site_ids])
        if self.body_id is not None:
            rotation = configuration.data.xmat[self.body_id].reshape(3, 3)
            points.extend(configuration.data.xpos[self.body_id] + self.offsets @ rotation.T)
        return np.mean(points, axis=0)

    def jacobian(self, configuration) -> np.ndarray:
        total = np.zeros((3, self.model.nv), dtype=float)
        count = 0
        jacr = np.zeros((3, self.model.nv), dtype=float)
        for site_id in self.site_ids:
            jacp = np.zeros((3, self.model.nv), dtype=float)
            mj.mj_jacSite(self.model, configuration.data, jacp, jacr, int(site_id))
            total += jacp
            count += 1
        if self.body_id is not None:
            rotation = configuration.data.xmat[self.body_id].reshape(3, 3)
            for offset in self.offsets:
                point = configuration.data.xpos[self.body_id] + rotation @ offset
                jacp = np.zeros((3, self.model.nv), dtype=float)
                mj.mj_jac(self.model, configuration.data, jacp, jacr, point, self.body_id)
                total += jacp
                count += 1
        return total / max(count, 1)


class TerrainPointContactTask(Task):
    def __init__(self, model, channel_specs: dict, normal_cost: float, tangent_cost: float, clearance: float) -> None:
        self.model = model
        self.channels = list(channel_specs)
        self.points = {name: RobotPointGroup(model, spec) for name, spec in channel_specs.items()}
        self.normal_cost = float(normal_cost)
        self.tangent_cost = float(tangent_cost)
        self.clearance = float(clearance)
        self.targets: dict[str, dict] = {}
        self.anchors: dict[str, np.ndarray] = {}
        self.previous_state = {name: "NONE" for name in self.channels}
        self.none_hold_steps = 3
        self.none_counts = {name: 0 for name in self.channels}
        costs = np.tile([self.normal_cost, self.tangent_cost, self.tangent_cost], len(self.channels))
        super().__init__(cost=costs, gain=0.55, lm_damping=1.0)

    def set_contacts(self, configuration, contacts: dict) -> None:
        targets = {}
        for name in self.channels:
            contact = contacts.get(name, {})
            score = float(np.clip(contact.get("score", 0.0), 0.0, 1.0))
            state = str(contact.get("state", "NONE"))
            # Keep one anchor for the whole contact episode.  A brief
            # STATIC/SLIDING classification change must not re-anchor after
            # the point has already moved along the surface.
            if state in {"STATIC", "SLIDING"} and name not in self.anchors:
                self.anchors[name] = self.points[name].point(configuration).copy()
                self.none_counts[name] = 0
            elif state == "NONE":
                self.none_counts[name] += 1
                if self.none_counts[name] >= self.none_hold_steps:
                    self.anchors.pop(name, None)
            else:
                self.none_counts[name] = 0
            targets[name] = {**contact, "score": score, "state": state}
            self.previous_state[name] = state
        self.targets = targets

    def _channel_data(self, name: str):
        target = self.targets.get(name, {})
        activation = float(target.get("score", 0.0)) if target.get("state", "NONE") != "NONE" else 0.0
        normal = np.asarray(target.get("surface_normal_solver", [0, 0, 1]), dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        surface = np.asarray(target.get("surface_point_solver", [0, 0, 0]), dtype=float)
        tangents = tangent_basis(normal)
        if target.get("state") == "STATIC" and name in self.anchors:
            anchor = self.anchors[name]
        else:
            anchor = np.asarray(target.get("human_point_solver", surface), dtype=float)
        return activation, normal, surface, tangents, anchor

    def compute_error(self, configuration) -> np.ndarray:
        residual = []
        for name in self.channels:
            point = self.points[name].point(configuration)
            activation, normal, surface, tangents, anchor = self._channel_data(name)
            residual.extend(activation * np.array([
                normal @ (point - surface) - self.clearance,
                tangents[:, 0] @ (point - anchor),
                tangents[:, 1] @ (point - anchor),
            ]))
        return np.asarray(residual)

    def compute_jacobian(self, configuration) -> np.ndarray:
        jacobian = np.zeros((3 * len(self.channels), self.model.nv))
        for index, name in enumerate(self.channels):
            activation, normal, _, tangents, _ = self._channel_data(name)
            point_jacobian = self.points[name].jacobian(configuration)
            jacobian[3 * index:3 * index + 3] = activation * np.vstack((normal, tangents.T)) @ point_jacobian
        return jacobian


class TerrainFootOrientationTask(Task):
    def __init__(self, model, foot_bodies: dict[str, str], local_normal: np.ndarray, cost: float) -> None:
        self.model = model
        self.body_ids = {side: model.body(name).id for side, name in foot_bodies.items()}
        self.local_normal = np.asarray(local_normal, dtype=float)
        self.local_normal /= max(float(np.linalg.norm(self.local_normal)), 1e-12)
        self.targets = {side: {"activation": 0.0, "normal": np.array([0, 0, 1.0])} for side in foot_bodies}
        super().__init__(cost=np.full(2 * len(foot_bodies), float(cost)), gain=0.45, lm_damping=1.0)

    def set_contacts(self, contacts: dict, flat_foot: dict) -> None:
        for side in self.body_ids:
            heel, toe = contacts.get(f"{side}_heel", {}), contacts.get(f"{side}_toe", {})
            activation = float(flat_foot.get(side, 0.0))
            normal = np.asarray(heel.get("surface_normal_solver", [0, 0, 1]), dtype=float)
            if activation > 0.0 and heel.get("surface_id") == toe.get("surface_id"):
                normal = normal + np.asarray(toe.get("surface_normal_solver", normal), dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            self.targets[side] = {"activation": activation, "normal": normal}

    def compute_error(self, configuration) -> np.ndarray:
        residual = []
        for side, body_id in self.body_ids.items():
            target = self.targets[side]
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            axis = rotation @ self.local_normal
            residual.extend(target["activation"] * tangent_basis(target["normal"]).T @ axis)
        return np.asarray(residual)

    def compute_jacobian(self, configuration) -> np.ndarray:
        jacobian = np.zeros((2 * len(self.body_ids), self.model.nv))
        for index, (side, body_id) in enumerate(self.body_ids.items()):
            target = self.targets[side]
            jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
            mj.mj_jacBody(self.model, configuration.data, jacp, jacr, body_id)
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            axis = rotation @ self.local_normal
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            jacobian[2 * index:2 * index + 2] = target["activation"] * tangent_basis(target["normal"]).T @ (-skew @ jacr)
        return jacobian


class FootFrameTask(TerrainFootOrientationTask):
    """Foot-frame orientation using measured heel-to-toe forward direction."""

    def __init__(self, model, foot_bodies: dict[str, str], local_normal: np.ndarray, cost: float) -> None:
        # TerrainFootOrientationTask has two residuals per foot; this task
        # adds two forward-direction residuals, so allocate the matching cost
        # vector before Mink validates task dimensions.
        self.model = model
        self.body_ids = {side: model.body(name).id for side, name in foot_bodies.items()}
        self.local_normal = np.asarray(local_normal, dtype=float)
        self.local_normal /= max(float(np.linalg.norm(self.local_normal)), 1e-12)
        self.targets = {side: {"activation": 0.0, "normal": np.array([0, 0, 1.0]), "forward": np.array([1.0, 0.0, 0.0])} for side in foot_bodies}
        super(TerrainFootOrientationTask, self).__init__(cost=np.full(4 * len(foot_bodies), float(cost)), gain=0.45, lm_damping=1.0)

    def set_contacts(self, contacts: dict, flat_foot: dict) -> None:
        super().set_contacts(contacts, flat_foot)
        for side in self.body_ids:
            heel = contacts.get(f"{side}_heel", {})
            toe = contacts.get(f"{side}_toe", {})
            if str(heel.get("state", "NONE")) == "NONE":
                self.targets[side]["activation"] = float(heel.get("airborne_activation", 0.15))
                self.targets[side]["normal"] = np.asarray(heel.get("human_foot_normal_solver", [0, 0, 1]), dtype=float)
                self.targets[side]["normal"] /= max(float(np.linalg.norm(self.targets[side]["normal"])), 1e-12)
            forward = np.asarray(toe.get("human_point_solver", [1, 0, 0]), dtype=float) - np.asarray(heel.get("human_point_solver", [0, 0, 0]), dtype=float)
            source_forward = np.asarray(heel.get("human_foot_forward_solver", forward), dtype=float)
            normal = self.targets[side]["normal"]
            if str(heel.get("state", "NONE")) == "NONE":
                forward = source_forward
            forward = forward - normal * float(forward @ normal)
            if np.linalg.norm(forward) > 1e-8:
                self.targets[side]["forward"] = forward / np.linalg.norm(forward)
            else:
                self.targets[side]["forward"] = np.array([1.0, 0.0, 0.0])

    def compute_error(self, configuration) -> np.ndarray:
        residual = []
        for side, body_id in self.body_ids.items():
            target = self.targets[side]
            activation = float(target["activation"])
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            axis = rotation @ self.local_normal
            tangent = tangent_basis(target["normal"])
            actual_forward = rotation[:, 0]
            residual.extend(activation * np.r_[tangent.T @ axis, tangent.T @ (actual_forward - target["forward"])])
        return np.asarray(residual)

    def compute_jacobian(self, configuration) -> np.ndarray:
        jacobian = np.zeros((4 * len(self.body_ids), self.model.nv))
        for index, (side, body_id) in enumerate(self.body_ids.items()):
            target = self.targets[side]
            jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
            mj.mj_jacBody(self.model, configuration.data, jacp, jacr, body_id)
            rotation = configuration.data.xmat[body_id].reshape(3, 3)
            tangent = tangent_basis(target["normal"])
            axis = rotation @ self.local_normal
            skew_axis = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            actual_forward = rotation[:, 0]
            skew_forward = np.array([[0, -actual_forward[2], actual_forward[1]], [actual_forward[2], 0, -actual_forward[0]], [-actual_forward[1], actual_forward[0], 0]])
            rows = np.vstack((tangent.T @ (-skew_axis @ jacr), tangent.T @ (-skew_forward @ jacr)))
            jacobian[4 * index:4 * index + 4] = target["activation"] * rows
        return jacobian


class TerrainFootTemporalTask(Task):
    CHANNELS = ("left_heel", "left_toe", "right_heel", "right_toe")

    def __init__(self, model, channel_specs: dict, dt: float, cost: float) -> None:
        self.model = model
        self.points = {name: RobotPointGroup(model, channel_specs[name]) for name in self.CHANNELS}
        self.dt = max(float(dt), 1e-9)
        self.contacts: dict[str, dict] = {}
        self.previous_output: dict[str, np.ndarray] = {}
        self.frame_previous: dict[str, np.ndarray] = {}
        super().__init__(cost=np.full(8, float(cost)), gain=0.45, lm_damping=1.0)

    def begin_frame(self, configuration, contacts: dict) -> None:
        self.contacts = contacts
        self.frame_previous = {
            name: self.previous_output.get(name, point.point(configuration)).copy()
            for name, point in self.points.items()
        }

    def end_frame(self, configuration) -> None:
        self.previous_output = {name: point.point(configuration).copy() for name, point in self.points.items()}

    def compute_error(self, configuration) -> np.ndarray:
        residual = []
        for name in self.CHANNELS:
            contact = self.contacts.get(name, {})
            activation = float(contact.get("score", 0.0)) if contact.get("state") == "STATIC" else 0.0
            tangents = tangent_basis(np.asarray(contact.get("surface_normal_solver", [0, 0, 1]), dtype=float))
            velocity = (self.points[name].point(configuration) - self.frame_previous[name]) / self.dt
            residual.extend(activation * tangents.T @ velocity)
        return np.asarray(residual)

    def compute_jacobian(self, configuration) -> np.ndarray:
        jacobian = np.zeros((8, self.model.nv))
        for index, name in enumerate(self.CHANNELS):
            contact = self.contacts.get(name, {})
            activation = float(contact.get("score", 0.0)) if contact.get("state") == "STATIC" else 0.0
            tangents = tangent_basis(np.asarray(contact.get("surface_normal_solver", [0, 0, 1]), dtype=float))
            jacobian[2 * index:2 * index + 2] = activation * tangents.T @ self.points[name].jacobian(configuration) / self.dt
        return jacobian
