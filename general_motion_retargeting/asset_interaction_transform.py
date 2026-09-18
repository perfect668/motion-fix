"""Conservative, asset-agnostic scene adaptation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class AssetInteractionTransform:
    """One constant similarity/anisotropic transform for a scene asset."""

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    scale_axis_1: float = 1.0
    scale_axis_2: float = 1.0
    scale_up: float = 1.0

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=float).reshape(3)
        scales = np.array([self.scale_axis_1, self.scale_axis_2, self.scale_up], dtype=float)
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("AssetInteractionTransform values must be finite and positive")
        object.__setattr__(self, "translation", translation)
        for name, value in zip(("scale_axis_1", "scale_axis_2", "scale_up"), scales):
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "yaw", float(self.yaw))

    @classmethod
    def from_config(cls, config: dict) -> "AssetInteractionTransform":
        values = {key: config.get(key, default) for key, default in (
            ("translation", [0.0, 0.0, 0.0]), ("yaw", 0.0),
            ("scale_axis_1", 1.0), ("scale_axis_2", 1.0), ("scale_up", 1.0),
        )}
        lower = float(config.get("scale_min", 0.85))
        upper = float(config.get("scale_max", 1.15))
        if lower <= 0.0 or upper < lower:
            raise ValueError("asset scale bounds must be positive and ordered")
        scales = [float(values[name]) for name in ("scale_axis_1", "scale_axis_2", "scale_up")]
        if any(value < lower or value > upper for value in scales):
            raise ValueError(f"asset scales {scales} are outside [{lower}, {upper}]")
        return cls(**values)

    @property
    def linear(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag(
            [self.scale_axis_1, self.scale_axis_2, self.scale_up]
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.linear.T + self.translation

    def transform_normals(self, normals: np.ndarray) -> np.ndarray:
        values = np.asarray(normals, dtype=float) @ np.linalg.inv(self.linear)
        return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)

    def transform_pose(self, pose: np.ndarray) -> np.ndarray:
        result = np.asarray(pose, dtype=float).reshape(4, 4).copy()
        matrix = np.eye(4)
        matrix[:3, :3] = self.linear
        matrix[:3, 3] = self.translation
        return matrix @ result

    def to_dict(self) -> dict:
        return {
            "translation": self.translation.tolist(), "yaw": self.yaw,
            "scale_axis_1": self.scale_axis_1, "scale_axis_2": self.scale_axis_2,
            "scale_up": self.scale_up,
        }

    def optimize(
        self,
        objective: Callable[["AssetInteractionTransform"], float],
        *,
        max_iterations: int = 3,
        translation_step: float = 0.02,
        yaw_step: float = 0.05,
        scale_step: float = 0.03,
        scale_min: float = 0.85,
        scale_max: float = 1.15,
    ) -> tuple["AssetInteractionTransform", float]:
        """Deterministically minimize an interaction objective by coordinates."""
        if max_iterations < 0 or scale_min <= 0.0 or scale_max < scale_min:
            raise ValueError("invalid asset optimization bounds")
        current = self
        best = float(objective(current))
        if not np.isfinite(best):
            raise ValueError("asset interaction objective must be finite")
        steps = np.array([
            translation_step, translation_step, translation_step, yaw_step,
            scale_step, scale_step, scale_step,
        ], dtype=float)
        if np.any(steps < 0.0):
            raise ValueError("asset optimization step sizes must be non-negative")

        for _ in range(int(max_iterations)):
            improved = False
            for index in range(7):
                for sign in (-1.0, 1.0):
                    values = np.array([
                        *current.translation.tolist(), current.yaw,
                        current.scale_axis_1, current.scale_axis_2, current.scale_up,
                    ])
                    values[index] += sign * steps[index]
                    values[4:7] = np.clip(values[4:7], scale_min, scale_max)
                    trial = replace(
                        current, translation=values[:3], yaw=values[3],
                        scale_axis_1=values[4], scale_axis_2=values[5], scale_up=values[6],
                    )
                    value = float(objective(trial))
                    if np.isfinite(value) and value < best:
                        current, best, improved = trial, value, True
                        break
            steps *= 0.5
            if not improved or float(np.max(steps)) < 1e-8:
                break
        return current, best
