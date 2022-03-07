from logging import raiseExceptions
import numpy as np
import jax
import jax.numpy as jnp
import pdb
import Pplan.utils.geometry_utils as GeoUtils
import matplotlib.pyplot as plt

STATE_INDEX = [0, 1, 2, 4]


def mean_control_effort_coefficients(x0, dx0, xf, dxf):
    """Returns `(c4, c3, c2)` corresponding to `c4 * tf**-4 + c3 * tf**-3 + c2 * tf**-2`."""
    return (12 * (x0 - xf) ** 2, 12 * (dx0 + dxf) * (x0 - xf), 4 * dx0 ** 2 + 4 * dx0 * dxf + 4 * dxf ** 2)


def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (x0, dx0, -2 * dx0 / tf - dxf / tf - 3 * x0 / tf ** 2 + 3 * xf / tf ** 2,
            dx0 / tf ** 2 + dxf / tf ** 2 + 2 * x0 / tf ** 3 - 2 * xf / tf ** 3)


def compute_interpolating_spline(state_0, state_f, tf):
    x0, y0, v0, q0 = state_0
    xf, yf, vf, qf = state_f
    dx0, dy0 = v0 * jnp.cos(q0), v0 * jnp.sin(q0)
    dxf, dyf = vf * jnp.cos(qf), vf * jnp.sin(qf)
    return (
        jnp.array(cubic_spline_coefficients(x0, dx0, xf, dxf, tf)),
        jnp.array(cubic_spline_coefficients(y0, dy0, yf, dyf, tf)),
        tf,
    )


def compute_spline_xyvaqrt(x_coefficients, y_coefficients, tf, N=10):
    t = jnp.linspace(0, tf, N)
    tp = t[:, None] ** np.arange(4)
    dtp = t[:, None] ** np.array([0, 0, 1, 2]) * np.arange(4)
    ddtp = t[:, None] ** np.array([0, 0, 0, 1]) * np.array([0, 0, 2, 6])
    vx = dtp @ x_coefficients
    vy = dtp @ y_coefficients
    v = jnp.hypot(vx, vy)
    v_pos = jnp.clip(v, 1e-4, None)
    ax = ddtp @ x_coefficients
    ay = ddtp @ y_coefficients
    a = (ax * vx + ay * vy) / v_pos
    r = (-ax * vy + ay * vx) / (v_pos ** 2)
    yaw = jnp.arctan2(vy, vx)
    return jnp.stack((
        tp @ x_coefficients,
        tp @ y_coefficients,
        v,
        a,
        yaw,
        r,
        t,
    ), -1)


def mean_control_effort_coefficients(x0, dx0, xf, dxf):
    """Returns `(c4, c3, c2)` corresponding to `c4 * tf**-4 + c3 * tf**-3 + c2 * tf**-2`."""
    return (12 * (x0 - xf) ** 2, 12 * (dx0 + dxf) * (x0 - xf), 4 * dx0 ** 2 + 4 * dx0 * dxf + 4 * dxf ** 2)


class SplinePlanner(object):
    def __init__(self, dx_grid=None, dy_grid=None, acce_grid=None, dyaw_grid=None, max_steer=0.5, max_rvel=8,
                 acce_bound=[-6, 4], vbound=[-10, 30], spline_order=3):
        self.spline_order = spline_order
        assert spline_order == 3
        if dx_grid is None:
            self.dx_grid = np.array([-8., -4., 0, 4., 8.])
        else:
            self.dx_grid = dx_grid
        if dy_grid is None:
            self.dy_grid = np.array([-4., -2., 0, 2., 4.])
        else:
            self.dy_grid = dy_grid
        if acce_grid is None:
            self.acce_grid = np.array([-1., -0.5, 0., 0.5, 1.])
        else:
            self.acce_grid = acce_grid
        if dyaw_grid is None:
            self.dyaw_grid = np.array([-np.pi / 12, 0, np.pi / 12])
        else:
            self.dyaw_grid = dyaw_grid
        self.max_steer = max_steer
        self.max_rvel = max_rvel
        self.acce_bound = acce_bound
        self.vbound = vbound

    def calc_trajectories(self, x0, tf, xf):
        if x0.ndim == 1:
            xc, yc, tf = jax.vmap(compute_interpolating_spline, in_axes=(None, 0, None))(x0, xf, tf)
        elif x0.ndim == xf.ndim:
            xc, yc, tf = jax.vmap(compute_interpolating_spline, in_axes=(0, 0, None))(x0, xf, tf)
        else:
            raise ValueError("wrong dimension for x0")
        traj = jax.vmap(compute_spline_xyvaqrt)(xc, yc, tf)
        return traj

    def gen_terminals(self, x0, tf):
        if x0.ndim == 1:
            delta_x = np.array([
                np.array([x, y, v, yaw]) for x in self.dx_grid[2:3] for y in self.dy_grid for v in
                self.acce_grid[0::2] * tf for yaw in self.dyaw_grid[1:2]
            ])
            xy = np.concatenate((delta_x[:, 0:1] + delta_x[:, 2:3] / 2 * tf + x0[2:3] * tf, delta_x[:, 1:2]), -1)
            rotated_xy = GeoUtils.batch_rotate_2D(xy, x0[3]) + x0[:2]
            return np.concatenate((rotated_xy, delta_x[:, 2:] + x0[2:]), -1) + x0[None, :]
        elif x0.ndim == 2:
            delta_x = np.array([
                np.array([x, y, v, yaw]) for x in self.dx_grid[2:3] for y in self.dy_grid for v in
                self.acce_grid[0::2] * tf for yaw in self.dyaw_grid[1:2]
            ])
            delta_x = np.tile(delta_x, [x0.shape[0], 1, 1])

            xy = np.concatenate(
                (delta_x[:, :, 0:1] + delta_x[:, :, 2:3] / 2 * tf + x0[:, None, 2:3] * tf, delta_x[:, :, 1:2]), -1)
            rotated_xy = GeoUtils.batch_rotate_2D(xy, x0[:, 3:4]) + x0[:, None, :2]

            return np.concatenate((rotated_xy, delta_x[:, :, 2:]), -1) + x0[:, None, :]
        else:
            raise Exceptions("x0 must have dimension 1 or 2")

    def feasible_flag(self, traj):
        feas_flag = ((traj[..., 2] >= self.vbound[0]) & (traj[..., 2] < self.vbound[1]) & \
                     (traj[..., 3] >= self.acce_bound[0]) & (traj[..., 3] <= self.acce_bound[1]) & \
                     (np.abs(traj[..., 5] * traj[..., 2]) <= self.max_rvel) & (
                             np.abs(traj[..., 2]) * self.max_steer >= np.abs(traj[..., 5]))).all(1)
        return feas_flag

    def gen_trajectories(self, x0, tf, dyn_filter=True):
        xf = self.gen_terminals(x0, tf)
        # x, y, v, a, yaw,r, t
        traj = jnp.asarray(self.calc_trajectories(x0, tf, xf))
        if dyn_filter:
            feas_flag = self.feasible_flag(traj)
            return traj[feas_flag, :], xf[feas_flag, :]
        else:
            return traj, xf

    def gen_trajectory_batch(self, x0_set, tf, dyn_filter=True):
        xf_set = self.gen_terminals(x0_set, tf)
        num_node = x0_set.shape[0]
        num = xf_set.shape[1]
        x0_tiled = np.tile(x0_set, [num, 1])
        xf_tiled = xf_set.reshape(-1, xf_set.shape[-1])
        traj = self.calc_trajectories(x0_tiled, tf, xf_tiled)
        if dyn_filter:
            feas_flag = self.feasible_flag(traj)
        else:
            feas_flag = np.ones(num * num_node, dtype=numpy.bool)
        feas_flag = feas_flag.reshape(num_node, num)
        traj = traj.reshape(num_node, num, *traj.shape[1:])
        return [traj[i, feas_flag[i]] for i in range(num_node)]

    def gen_trajectory_tree(self, x0, tf, n_layers, dyn_filter=True):
        trajs = list()
        nodes = [x0[None, :]]
        for i in range(n_layers):
            xf = self.gen_terminals(nodes[i], tf)
            x0i = np.tile(nodes[i], [xf.shape[1], 1])
            xf = xf.reshape(-1, xf.shape[-1])

            traj = self.calc_trajectories(x0i, tf, xf)
            if dyn_filter:
                feas_flag = self.feasible_flag(traj)
                traj = traj[feas_flag]
                xf = xf[feas_flag]

            trajs.append(traj)

            nodes.append(xf.reshape(-1, xf.shape[-1]))
        return trajs, nodes[1:]


if __name__ == "__main__":
    planner = SplinePlanner()
    x0 = np.array([1., 2., 10., 0.])
    tf = 5
    traj, xf = planner.gen_trajectories(x0, tf)
    trajs = planner.gen_trajectory_batch(xf, tf)
    # # x, y, v, a, yaw,r, t = traj
    # msize = 12
    # trajs, nodes = planner.gen_trajectory_tree(x0, tf, 2)
    # plt.figure(figsize=(20, 10))
    # plt.plot(x0[0], x0[1], marker="o", color="b", markersize=msize)
    # for node, traj in zip(nodes, trajs):
    #     x = traj[..., 0]
    #     y = traj[..., 1]
    #     plt.plot(x.T, y.T, color="k")
    #     for p in node:
    #         plt.plot(p[0], p[1], marker="o", color="b", markersize=msize)
    # plt.show()
