"""Constrained pose search on an independent MuJoCo planning state."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class IKError(RuntimeError):
    pass


class PoseSolver:
    """A five-joint solver; incompatible six-dimensional goals are rejected."""

    def __init__(self, env):
        self.model = env.model
        self.probe = mujoco.MjData(self.model)
        self.probe.qpos[:] = env.data.qpos
        self.qids = env.qpos_ids[:5]
        self.site = env.tcp_id
        self.bounds = env.control_bounds[:5].copy()

    def pose(self, q):
        self.probe.qpos[self.qids] = q
        mujoco.mj_kinematics(self.model, self.probe)
        return (self.probe.site_xpos[self.site].copy(),
                self.probe.site_xmat[self.site].reshape(3, 3).copy())

    def solve(self, position, rotation, seed, *, position_tolerance=.003,
              angle_tolerance=.10):
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)

        def residual(q):
            p, r = self.pose(q)
            angle = Rotation.from_matrix(rotation @ r.T).as_rotvec()
            return np.r_[p - position, .07 * angle]

        best = None
        # Nearby solutions preserve the elbow branch. A small template bank
        # also handles reset poses; no solution is forced into acceptance.
        seeds = [np.asarray(seed)[:5],
                 [seed[0], -.5, 1.0, 1.0, seed[4]],
                 [seed[0], -.5, 1.0, -.5, seed[4]]]
        for candidate in seeds:
            result = least_squares(residual,
                np.clip(candidate, self.bounds[:, 0]+1e-6, self.bounds[:, 1]-1e-6),
                bounds=(self.bounds[:, 0]+1e-7, self.bounds[:, 1]-1e-7),
                max_nfev=180, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            p, r = self.pose(result.x)
            p_error = float(np.linalg.norm(p-position))
            a_error = float(np.linalg.norm(Rotation.from_matrix(rotation @ r.T).as_rotvec()))
            score = p_error + .07*a_error
            if best is None or score < best[0]:
                best = (score, result.x.copy(), p_error, a_error)
            if p_error <= position_tolerance and a_error <= angle_tolerance:
                return result.x.copy()
        raise IKError(f"Unreachable pose: position_error={best[2]:.5f}m, angle_error={best[3]:.4f}rad; "
                      f"goal={position.round(4).tolist()}")


def grasp_rotation(brush_axis):
    """Approach down; tool Y follows the horizontal toothbrush long axis."""
    x = np.array([0., 0., -1.])
    y = np.asarray(brush_axis, dtype=float).copy()
    y[2] = 0
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def upright_rotation(position):
    """Tool X points radially outward; tool Y (held brush long axis) points up."""
    x = np.array([position[0]-.0388353, position[1], 0.])
    x /= np.linalg.norm(x)
    y = np.array([0., 0., 1.])
    return np.column_stack([x, y, np.cross(x, y)])
