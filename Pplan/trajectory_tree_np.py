import numpy as np
import itertools
import pdb
import matplotlib.pyplot as plt


STATE_INDEX = [0, 1, 2, 4]


class TrajTree(object):

    def __init__(self, traj, parent, time):
        self.traj = traj
        self.state = traj[-1, STATE_INDEX]
        self.children = list()
        self.parent = parent
        self.time = time
        if parent is not None:
            self.total_traj = np.concatenate((parent.total_traj, traj), 0)
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
            children_set = [child for child in self.children if not child.isleaf()]
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
        for i in range(n_steps):
            nodes_state = np.array([node.state for node in nodes])
            trajs = expand_func(nodes_state)
            new_nodes = list()
            for node, traj in zip(nodes, trajs):
                node.children = [TrajTree(traj[i], node, node.time + 1) for i in range(traj.shape[0])]
                new_nodes += node.children
            nodes = new_nodes

    def plot_tree(self, ax=None, msize=12):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.state[0], self.state[1], marker="o", color="b", markersize=msize)
        if self.traj.shape[0] > 1:
            ax.plot(self.traj[:, 0], self.traj[:, 1], color="k")
        for child in self.children:
            child.plot_tree(ax)


if __name__ == "__main__":
    from Pplan.spline_planner_jax import SplinePlanner
    from Pplan.utils.timer import Timer
    timer = Timer()

    planner = SplinePlanner()
    timer.tic()
    traj = np.array([[1., 1., 5., 0., 0., 0., 0.]])
    x0 = TrajTree(traj, None, 0)
    tf = 5
    expand_func = lambda x: planner.gen_trajectory_batch(x, tf)
    x0.grow_tree(expand_func, 2)
    t = timer.toc()
    print("execution time:",t)
    # x0.plot_tree()
    # plt.show()
    # pdb.set_trace()
