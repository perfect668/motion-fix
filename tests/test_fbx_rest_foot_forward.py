import importlib.util
from pathlib import Path

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "fbx_entry", Path(__file__).parents[1] / "scripts" / "fbx_to_canonical_npz.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
rest_foot_forward = _module.rest_foot_forward


def test_fbx_rest_foot_forward():
    # Bind axis is Y while local X is intentionally unrelated.
    axis = rest_foot_forward([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], np.eye(3))
    assert np.allclose(axis, [0.0, 1.0, 0.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(
        rest_foot_forward([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], rotation), [-1.0, 0.0, 0.0]
    )
