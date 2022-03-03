STATE_INDEX = [0,1,2,4]

class TrajTree(object):
    def __init__(self,traj):
        self.traj = traj
        self.state =traj[-1]
