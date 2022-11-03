import itertools
from collections import defaultdict
class Tree(object):

    def __init__(self, content, parent, depth):
        self.content = content
        self.children = list()
        self.parent = parent
        if parent is not None:
            parent.expand(self)
        self.depth = depth
        self.attribute = dict()

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


    def get_all_leaves(self,leaf_set=[]):
        if self.isleaf():
            leaf_set.append(self)
        else:
            for child in self.children:
                leaf_set = child.get_all_leaves(leaf_set)
        return leaf_set

    @staticmethod
    def get_nodes_by_level(obj,depth,nodes=None,trim_short_branch=True):
        assert obj.depth<=depth
        if nodes is None:
            nodes = defaultdict(lambda: list())
        if obj.depth==depth:
            nodes[depth].append(obj)
            return nodes, True
        else:
            if obj.isleaf():
                return nodes, False

            else:
                flag = False
                children_flags = dict()
                for child in obj.children:
                    nodes, child_flag = Tree.get_nodes_by_level(child,depth,nodes)
                    children_flags[child] = child_flag
                    flag = flag or child_flag
                if trim_short_branch:
                    obj.children = [child for child in obj.children if children_flags[child]]
                if flag:
                    nodes[obj.depth].append(obj)
                return nodes, flag

    @staticmethod
    def get_children(obj):
        if isinstance(obj, Tree):
            return obj.children
        elif isinstance(obj, list):
            children = [node.children for node in obj]
            children = list(itertools.chain.from_iterable(children))
            return children
        else:
            raise TypeError("obj must be a TrajTree or a list")