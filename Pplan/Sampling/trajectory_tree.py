import torch
import itertools
import pdb
import matplotlib.pyplot as plt
import numpy as np
from Pplan.Sampling.tree import Tree

STATE_INDEX = [0, 1, 2, 4]



class TrajTree(Tree):

    def __init__(self, traj, parent, depth):
        self.traj = traj
        self.state = traj[-1, STATE_INDEX]
        self.children = list()
        self.parent = parent
        self.depth = depth
        self.attribute = dict()
        if parent is not None:
            self.total_traj = torch.cat((parent.total_traj, traj), 0)
            parent.expand(self)
        else:
            self.total_traj = traj

    def expand_children(self, expand_func):
        trajs = expand_func(self.state)
        children = [TrajTree(traj, self, self.depth + 1) for traj in trajs]
        self.expand_set(children)


    # @staticmethod
    # def get_nodes_by_level(obj,depth,nodes=None):
    #     assert obj.depth<=depth
    #     if nodes is None:
    #         nodes = defaultdict(lambda: list())
    #     if obj.depth==depth:
    #         nodes[depth].append(obj)
    #         return nodes, True
    #     else:
    #         if obj.isleaf():
    #             return nodes, False
    #         else:
    #             flag = False
    #             for child in obj.children:
    #                 nodes, child_flag = TrajTree.get_nodes_by_level(child,depth,nodes)
    #                 flag = flag or child_flag
    #             if flag:
    #                 nodes[obj.depth].append(obj)
    #             return nodes, flag

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
        with torch.no_grad():
            for i in range(n_steps):
                nodes_state = torch.stack(
                    [node.state for node in nodes], 0).to(device)
                trajs = expand_func(nodes_state)
                new_nodes = list()
                for node, traj in zip(nodes, trajs):
                    traj[..., -1] += node.traj[-1, -1]
                    node.children = [TrajTree(traj[i], node, node.depth + 1)
                                     for i in range(traj.shape[0])]
                    new_nodes += node.children
                nodes = new_nodes
                if len(new_nodes) == 0:
                    return

    def plot_tree(self, ax=None, msize=12):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        state = self.state.cpu().detach().numpy()
        
        ax.plot(state[0], state[1], marker="o", color="b", markersize=msize)
        if self.traj.shape[0] > 1:
            if self.parent is not None:
                traj_l = torch.cat((self.parent.traj[-1:],self.traj),0)
                traj = traj_l.cpu().detach().numpy()
            else:
                traj = self.traj.cpu().detach().numpy()
            ax.plot(traj[:, 0], traj[:, 1], color="k")
        for child in self.children:
            child.plot_tree(ax)
        return ax


if __name__ == "__main__":
    from Pplan.Sampling.spline_planner import SplinePlanner
    from Pplan.utils.timer import Timer

    planner = SplinePlanner("cuda")
    timer = Timer()
    timer.tic()
    with torch.no_grad():
        traj = torch.tensor([[4., 1., 5., 0., 0., 0., 0.]]).cuda()
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
            x, tf, (lane0, lane1, lane2),max_children=10)
        # def expand_func(x): return planner.gen_trajectory_batch(
        #     x, tf)
        x0.grow_tree(expand_func, 2)
        t = timer.toc()
        nodes, _ = TrajTree.get_nodes_by_level(x0,2)
        import pdb
        pdb.set_trace()
        ax = x0.plot_tree()
        ax.plot(lane0[:, 0], lane0[:, 1],
                linestyle='dashed', linewidth=3, color="r")
        ax.plot(lane1[:, 0], lane1[:, 1],
                linestyle='dashed', linewidth=3, color="r")
        ax.plot(lane2[:, 0], lane2[:, 1],
                linestyle='dashed', linewidth=3, color="r")
        plt.show()
    print("execution time:", t)
