from logging import raiseExceptions
import numpy as np
import torch
import pdb
import Pplan.utils.geometry_utils as GeoUtils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


STATE_INDEX = [0, 1, 2, 4]

device = "cuda" if torch.cuda.is_available() else "cpu"


def mean_control_effort_coefficients(x0, dx0, xf, dxf):
    """Returns `(c4, c3, c2)` corresponding to `c4 * tf**-4 + c3 * tf**-3 + c2 * tf**-2`."""
    return (12 * (x0 - xf) ** 2, 12 * (dx0 + dxf) * (x0 - xf), 4 * dx0 ** 2 + 4 * dx0 * dxf + 4 * dxf ** 2)


def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (x0, dx0, -2 * dx0 / tf - dxf / tf - 3 * x0 / tf ** 2 + 3 * xf / tf ** 2,
            dx0 / tf ** 2 + dxf / tf ** 2 + 2 * x0 / tf ** 3 - 2 * xf / tf ** 3)


def compute_interpolating_spline(state_0, state_f, tf):
    dx0, dy0 = state_0[..., 2] * \
        torch.cos(state_0[..., 3]), state_0[..., 2] * \
        torch.sin(state_0[..., 3])
    dxf, dyf = state_f[..., 2] * \
        torch.cos(state_f[..., 3]), state_f[..., 2] * \
        torch.sin(state_f[..., 3])
    tf = tf * torch.ones_like(state_0[..., 0])
    return (
        torch.stack(cubic_spline_coefficients(
            state_0[..., 0], dx0, state_f[..., 0], dxf, tf), -1),
        torch.stack(cubic_spline_coefficients(
            state_0[..., 1], dy0, state_f[..., 1], dyf, tf), -1),
        tf,
    )


def compute_spline_xyvaqrt(x_coefficients, y_coefficients, tf, N=10):
    t = torch.arange(N).unsqueeze(0).to(tf.device) * tf.unsqueeze(-1) / (N - 1)
    tp = t[..., None] ** torch.arange(4).to(tf.device)
    dtp = t[..., None] ** torch.tensor([0, 0, 1, 2]
                                       ).to(tf.device) * torch.arange(4).to(tf.device)
    ddtp = t[..., None] ** torch.tensor([0, 0, 0, 1]).to(
        tf.device) * torch.tensor([0, 0, 2, 6]).to(tf.device)
    x_coefficients = x_coefficients.unsqueeze(-1)
    y_coefficients = y_coefficients.unsqueeze(-1)
    vx = dtp @ x_coefficients
    vy = dtp @ y_coefficients
    v = torch.hypot(vx, vy)
    v_pos = torch.clip(v, min=1e-4)
    ax = ddtp @ x_coefficients
    ay = ddtp @ y_coefficients
    a = (ax * vx + ay * vy) / v_pos
    r = (-ax * vy + ay * vx) / (v_pos ** 2)
    yaw = torch.atan2(vy, vx)
    return torch.cat((
        tp @ x_coefficients,
        tp @ y_coefficients,
        v,
        a,
        yaw,
        r,
        t.unsqueeze(-1),
    ), -1)


def mean_control_effort_coefficients(x0, dx0, xf, dxf):
    """Returns `(c4, c3, c2)` corresponding to `c4 * tf**-4 + c3 * tf**-3 + c2 * tf**-2`."""
    return (12 * (x0 - xf) ** 2, 12 * (dx0 + dxf) * (x0 - xf), 4 * dx0 ** 2 + 4 * dx0 * dxf + 4 * dxf ** 2)


class SplinePlanner(object):
    def __init__(self, device, dx_grid=None, dy_grid=None, acce_grid=None, dyaw_grid=None, max_steer=0.5, max_rvel=8,
                 acce_bound=[-6, 4], vbound=[-10, 30], spline_order=3,N_seg = 10):
        self.spline_order = spline_order
        self.device = device
        assert spline_order == 3
        if dx_grid is None:
            # self.dx_grid = torch.tensor([-4., 0, 4.]).to(self.device)
            self.dx_grid = torch.tensor([0.]).to(self.device)
        else:
            self.dx_grid = dx_grid
        if dy_grid is None:
            self.dy_grid = torch.tensor([-4., -2., 0, 2., 4.]).to(self.device)
        else:
            self.dy_grid = dy_grid
        if acce_grid is None:
            # self.acce_grid = torch.tensor([-1., -0.5, 0., 0.5, 1.]).to(self.device)
            self.acce_grid = torch.tensor([-1., 0., 1.]).to(self.device)
        else:
            self.acce_grid = acce_grid
        if dyaw_grid is None:
            self.dyaw_grid = torch.tensor(
                [-np.pi / 12, 0, np.pi / 12]).to(self.device)
        else:
            self.dyaw_grid = dyaw_grid
        self.max_steer = max_steer
        self.max_rvel = max_rvel
        self.acce_bound = acce_bound
        self.vbound = vbound
        self.N_seg = N_seg

    def calc_trajectories(self, x0, tf, xf):
        if x0.ndim == 1:
            x0_tile = x0.tile(xf.shape[0], 1)
            xc, yc, tf = compute_interpolating_spline(x0_tile, xf, tf)
        elif x0.ndim == xf.ndim:
            xc, yc, tf = compute_interpolating_spline(x0, xf, tf)
        else:
            raise ValueError("wrong dimension for x0")
        traj = compute_spline_xyvaqrt(xc, yc, tf, self.N_seg)
        return traj

    def gen_terminals_lane(self, x0, tf, lanes):
        if lanes is None or len(lanes)==0:
            return self.gen_terminals(x0, tf)

        gs = [self.dx_grid.shape[0], self.acce_grid.shape[0]]
        dx = self.dx_grid[:, None, None, None].repeat(1, 1, gs[1], 1).flatten()
        dv = self.acce_grid[None, None, :, None].repeat(
            gs[0], 1, 1, 1).flatten()*tf
        delta_x = list()
        if x0.ndim == 1:
            for lane in lanes:
                f, p_start = lane
                p_start = torch.from_numpy(p_start).to(x0.device)
                offset = x0[:2]-p_start[:2]
                s_offset = offset[0] * \
                    torch.cos(p_start[2])+offset[1]*torch.sin(p_start[2])
                ds = dx+dv/2*tf+x0[2:3]*tf
                ss = ds + s_offset
                xyyaw = torch.from_numpy(f(ss.cpu().numpy())).type(
                    torch.float).to(x0.device)
                delta_x.append(
                    torch.cat((xyyaw[:, :2], dv.reshape(-1, 1)+x0[2:3], xyyaw[:, 2:]), -1))
        elif x0.ndim == 2:
            for lane in lanes:
                f, p_start = lane
                p_start = torch.from_numpy(p_start).to(x0.device)
                offset = x0[:, :2]-p_start[None, :2]
                s_offset = offset[:, 0] * \
                    torch.cos(p_start[2])+offset[:, 1]*torch.sin(p_start[2])
                ds = (dx+dv/2*tf).unsqueeze(0)+x0[:, 2:3]*tf
                ss = ds + s_offset.unsqueeze(-1)
                xyyaw = torch.from_numpy(f(ss.cpu().numpy())).type(
                    torch.float).to(x0.device)
                delta_x.append(torch.cat((xyyaw[..., :2], dv.tile(
                    x0.shape[0], 1).unsqueeze(-1)+x0[:, None, 2:3], xyyaw[..., 2:]), -1))
        else:
            raise ValueError("x0 must have dimension 1 or 2")
        delta_x = torch.cat(delta_x, -2)
        return delta_x

    def gen_terminals(self, x0, tf):
        gs = [self.dx_grid.shape[0], self.dy_grid.shape[0],
              self.acce_grid.shape[0], self.dyaw_grid.shape[0]]
        dx = self.dx_grid[:, None, None, None].repeat(
            1, gs[1], gs[2], gs[3]).flatten()
        dy = self.dy_grid[None, :, None, None].repeat(
            gs[0], 1, gs[2], gs[3]).flatten()
        dv = tf * self.acce_grid[None, None, :,
                                 None].repeat(gs[0], gs[1], 1, gs[3]).flatten()
        dyaw = self.dyaw_grid[None, None, None, :].repeat(
            gs[0], gs[1], gs[2], 1).flatten()
        delta_x = torch.stack([dx, dy, dv, dyaw], -1)

        if x0.ndim == 1:
            xy = torch.cat(
                (delta_x[:, 0:1] + delta_x[:, 2:3] / 2 * tf + x0[2:3] * tf, delta_x[:, 1:2]), -1)
            rotated_xy = GeoUtils.batch_rotate_2D(xy, x0[3]) + x0[:2]
            return torch.cat((rotated_xy, delta_x[:, 2:] + x0[2:]), -1) + x0[None, :]
        elif x0.ndim == 2:

            delta_x = torch.tile(delta_x, [x0.shape[0], 1, 1])
            xy = torch.cat(
                (delta_x[:, :, 0:1] + delta_x[:, :, 2:3] / 2 * tf + x0[:, None, 2:3] * tf, delta_x[:, :, 1:2]), -1)
            rotated_xy = GeoUtils.batch_rotate_2D(
                xy, x0[:, 3:4]) + x0[:, None, :2]

            return torch.cat((rotated_xy, delta_x[:, :, 2:] + x0[:, None, 2:]), -1) + x0[:, None, :]
        else:
            raise ValueError("x0 must have dimension 1 or 2")

    def feasible_flag(self, traj):
        feas_flag = ((traj[..., 2] >= self.vbound[0]) & (traj[..., 2] < self.vbound[1]) &
                     (traj[..., 3] >= self.acce_bound[0]) & (traj[..., 3] <= self.acce_bound[1]) &
                     (torch.abs(traj[..., 5] * traj[..., 2]) <= self.max_rvel) & (
            torch.abs(traj[..., 2]) * self.max_steer >= torch.abs(traj[..., 5]))).all(1)
        return feas_flag

    def gen_trajectories(self, x0, tf, lanes=None, dyn_filter=True):
        if lanes is None:
            xf = self.gen_terminals(x0, tf)
        else:
            lane_interp = [GeoUtils.interp_lanes(lane) for lane in lanes]

            xf = self.gen_terminals_lane(
                x0, tf, lane_interp)
        
        # x, y, v, a, yaw,r, t
        traj = self.calc_trajectories(x0, tf, xf)
        if dyn_filter:
            feas_flag = self.feasible_flag(traj)
            return traj[feas_flag, 1:,:], xf[feas_flag]
        else:
            return traj[...,1:,:], xf

    def gen_trajectory_batch(self, x0_set, tf, lanes=None, dyn_filter=True):
        if lanes is None:
            xf_set = self.gen_terminals(x0_set, tf)
        else:
            lane_interp = [GeoUtils.interp_lanes(lane) for lane in lanes]
            xf_set = self.gen_terminals_lane(x0_set, tf, lane_interp)
                
        num_node = x0_set.shape[0]
        num = xf_set.shape[1]
        x0_tiled = torch.tile(x0_set, [num, 1])
        xf_tiled = xf_set.reshape(-1, xf_set.shape[-1])
        traj = self.calc_trajectories(x0_tiled, tf, xf_tiled)
        if dyn_filter:
            feas_flag = self.feasible_flag(traj)
        else:
            feas_flag = torch.ones(
                num * num_node, dtype=torch.bool).to(x0_set.device)
        feas_flag = feas_flag.reshape(num_node, num)
        traj = traj.reshape(num_node, num, *traj.shape[1:])
        return [traj[i, feas_flag[i],1:] for i in range(num_node)]

    def gen_trajectory_tree(self, x0, tf, n_layers, dyn_filter=True):
        trajs = list()
        nodes = [x0[None, :]]
        for i in range(n_layers):
            xf = self.gen_terminals(nodes[i], tf)
            x0i = torch.tile(nodes[i], [xf.shape[1], 1])
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
    planner = SplinePlanner("cuda")
    x0 = torch.tensor([1., 2., 10., 0.]).cuda()
    tf = 5
    traj, xf = planner.gen_trajectories(x0, tf)
    trajs = planner.gen_trajectory_batch(xf, tf)
    pdb.set_trace()
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
