import mujoco as mj
import numpy as np

from mink.tasks.task import Task


class FootSupportTask(Task):
    """Drive the bottoms of a rigid foot's support geoms to the ground plane."""

    def __init__(self, model, geom_names, ground_height=0.0, cost=100.0, gain=1.0):
        self.model = model
        self.geom_ids = np.asarray([model.geom(name).id for name in geom_names], dtype=int)
        self.body_ids = np.asarray(model.geom_bodyid[self.geom_ids], dtype=int)
        self.ground_height = float(ground_height)
        super().__init__(
            cost=np.full(len(self.geom_ids), float(cost)),
            gain=float(gain),
            lm_damping=0.0,
        )

    def compute_error(self, configuration):
        centers_z = configuration.data.geom_xpos[self.geom_ids, 2]
        radii = self.model.geom_size[self.geom_ids, 0]
        return centers_z - radii - self.ground_height

    def compute_jacobian(self, configuration):
        jacobian = np.zeros((len(self.geom_ids), self.model.nv))
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for row, (geom_id, body_id) in enumerate(zip(self.geom_ids, self.body_ids)):
            jacp.fill(0.0)
            jacr.fill(0.0)
            mj.mj_jac(
                self.model,
                configuration.data,
                jacp,
                jacr,
                configuration.data.geom_xpos[geom_id],
                int(body_id),
            )
            jacobian[row] = jacp[2]
        return jacobian

    def frozen_copy(self):
        names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, int(i)) for i in self.geom_ids]
        return FootSupportTask(
            self.model,
            names,
            ground_height=self.ground_height,
            cost=float(self.cost[0]),
            gain=0.0,
        )


class FootClearanceTask(Task):
    """Lift only the lowest sole point to a clearance plane."""

    def __init__(self, model, geom_names, clearance_height=0.002, cost=200.0):
        self.model = model
        self.geom_ids = np.asarray([model.geom(name).id for name in geom_names], dtype=int)
        self.body_ids = np.asarray(model.geom_bodyid[self.geom_ids], dtype=int)
        self.clearance_height = float(clearance_height)
        self._active_index = 0
        super().__init__(cost=np.array([float(cost)]), gain=1.0, lm_damping=0.0)

    def sole_heights(self, configuration):
        return (
            configuration.data.geom_xpos[self.geom_ids, 2]
            - self.model.geom_size[self.geom_ids, 0]
        )

    def compute_error(self, configuration):
        heights = self.sole_heights(configuration)
        self._active_index = int(np.argmin(heights))
        return np.array([heights[self._active_index] - self.clearance_height])

    def compute_jacobian(self, configuration):
        geom_id = int(self.geom_ids[self._active_index])
        body_id = int(self.body_ids[self._active_index])
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jac(
            self.model,
            configuration.data,
            jacp,
            jacr,
            configuration.data.geom_xpos[geom_id],
            body_id,
        )
        return jacp[2:3]
