import torch
import itertools
import pdb
import matplotlib.pyplot as plt
import numpy as np

STATE_INDEX = [0, 1, 2, 4]


class TrajTree(object):

    def __init__(self, traj, parent, time):
        self.traj = traj
        self.state = traj[-1, STATE_INDEX]
        self.children = list()
        self.parent = parent
        self.time = time
        if parent is not None:
            self.total_traj = torch.cat((parent.total_traj, traj), 0)
        else:
            self.total_traj = traj

    def expand(self, child):
        self.children.append(child)

    def expand_set(self, children):
        self.children += children

    def isroot(self):
        return self.parent is None

    def isleaf(self):
        return len(self.children) == 0

    def get_subseq_trajs(self):
        return [child.traj for child in self.children]

    def expand_children(self, expand_func):
        trajs = expand_func(self.state)
        children = [TrajTree(trajs[i], self, self.time + 1)]
        self.expand_set(children)

    def get_all_leaves(self):
        leaf_set = list()
        children_set = [self]
        while len(children_set) > 0:
            leaf_set += [child for child in self.children if child.isleaf()]
            children_set = [
                child for child in self.children if not child.isleaf()]
            children_set = [child.children for child in children_set]
            children_set = list(itertools.chain.from_iterable(children_set))
        return leaf_set

    @staticmethod
    def get_children(obj):
        if isinstance(obj, TrajTree):
            return obj.children
        elif isinstance(obj, list):
            children = [node.children for node in obj]
            children = list(itertools.chain.from_iterable(children))
            return children
        else:
            raise TypeError("obj must be a TrajTree or a list")

    def grow_tree(self, expand_func, n_steps):
        assert len(self.children) == 0
        nodes = [self]
        device = self.traj.device
        for i in range(n_steps):
            nodes_state = torch.stack(
                [node.state for node in nodes], 0).to(device)
            trajs = expand_func(nodes_state)
            new_nodes = list()
            for node, traj in zip(nodes, trajs):
                node.children = [TrajTree(traj[i], node, node.time + 1)
                                 for i in range(traj.shape[0])]
                new_nodes += node.children
            nodes = new_nodes

    def plot_tree(self, ax=None, msize=12):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        state = self.state.cpu().detach().numpy()
        traj = self.traj.cpu().detach().numpy()
        ax.plot(state[0], state[1], marker="o", color="b", markersize=msize)
        if traj.shape[0] > 1:
            ax.plot(traj[:, 0], traj[:, 1], color="k")
        for child in self.children:
            child.plot_tree(ax)
        return ax


if __name__ == "__main__":
    from Pplan.spline_planner import SplinePlanner
    from Pplan.utils.timer import Timer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    planner = SplinePlanner(device)
    timer = Timer()
    timer.tic()
    with torch.no_grad():
        traj = torch.tensor([[4., 1., 5., 0., 0., 0., 0.]]).to(device)
        lane0 = np.stack(
            [np.linspace(0, 100, 20), np.ones(20) * -3.6, np.zeros(20)]).T
        lane1 = np.stack(
            [np.linspace(0, 100, 20), np.ones(20) * 0, np.zeros(20)]).T
        theta = np.linspace(0, np.pi/2, 20)
        R = 50
        lane2 = np.stack(
            [R*np.sin(theta), R*(1-np.cos(theta))+3.6, theta]).T
        x0 = TrajTree(traj, None, 0)
        tf = 5

        def expand_func(x): return planner.gen_trajectory_batch(
            x, tf, (lane0, lane1, lane2))
        x0.grow_tree(expand_func, 1)
        t = timer.toc()

        ax = x0.plot_tree()
        ax.plot(lane0[:, 0], lane0[:, 1],
                linestyle='dashed', linewidth=3, color="r")
        ax.plot(lane1[:, 0], lane1[:, 1],
                linestyle='dashed', linewidth=3, color="r")
        ax.plot(lane2[:, 0], lane2[:, 1],
                linestyle='dashed', linewidth=3, color="r")
        plt.show()
    print("execution time:", t)
