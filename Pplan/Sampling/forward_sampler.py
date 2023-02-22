from logging import raiseExceptions
import numpy as np
import torch
import pdb
import Pplan.utils.geometry_utils as GeoUtils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
class ForwardSampler(object):
    def __init__(self,dt:float, acce_grid:list,dhm_grid:list, dhf_grid:list,  max_rvel=8,max_steer=0.5, vbound=[-5.0, 30],device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.accels = torch.tensor(acce_grid,device=self.device)
        self.dhf_grid = torch.tensor(dhf_grid,device=self.device)
        self.dhm_grid = torch.tensor(dhm_grid,device=self.device)
        self.max_rvel = max_rvel
        self.vbound = vbound
        self.max_steer = max_steer
        self.dt = dt

        
    def velocity_plan(self,x0:torch.Tensor,T:int,acce:Optional[torch.Tensor]=None):
        """plan velocity profile

        Args:
            x0 (torch.Tensor): [B, 4], X,Y,v,heading
            T (int): time horizon
            acce (torch.Tensor): [B, N]
        """
        bs = x0.shape[0]
        if acce is None:
            acce = self.accels[None,:].repeat_interleave(bs,0)
        v0 = x0[...,2] # [B]
        vdes = v0[:,None,None]+torch.arange(T,device=self.device)[None,None]*acce[:,:,None]*self.dt
        vplan = torch.clip(vdes,min=self.vbound[0],max=self.vbound[1])
        return vplan # [B, N, T]

    def lateral_plan(self,x0:torch.Tensor,vplan:torch.Tensor,dhf:torch.Tensor,dhm:torch.Tensor,T:int,bangbang=True):
        """plan lateral profile, 
            steering plan that ends with the desired heading change with mean heading change equal to dhm, if feasible
        Args:
            x0 (torch.Tensor): [B, 4], X,Y,v,heading
            vplan (torch.Tensor): [B, N, T] velocity profile 
            dhf (torch.Tensor): [B, M] desired heading change at the end of the horizon
            dhm (torch.Tensor): [B, M] mean heading change at the end of the horizon
            T (int): horizon
        """
        # using a linear steering profile
        bs,M = dhf.shape
        N = vplan.shape[1]
        vplan = vplan[:,:,None] # [B, N, 1, T]
        vl = torch.cat([x0[:,2].reshape(-1,1,1,1).repeat_interleave(N,1),vplan[...,:-1]],-1)
        acce = vplan-vl
        
        c0 = torch.abs(vl)
        c1 = torch.cumsum(c0*self.dt,-1)
        c2 = torch.cumsum(c1*self.dt,-1)
        c3 = torch.cumsum(c2*self.dt,-1)
                
        
        # algebraic equation: c1[T]*a0+c2[T]*a1 = dhf, c2[T]*a0+c3[T]*a1 = dhm
        
        a0 = (c3[...,-1]*dhf.unsqueeze(1)-c2[...,-1]*dhm.unsqueeze(1))/(c1[...,-1]*c3[...,-1]-c2[...,-1]**2) # [B, N, M]
        a1 = (dhf.unsqueeze(1)-c1[...,-1]*a0)/c2[...,-1]
        
        yawrate = a0[...,None]*c0+a1[...,None]*c1

        if bangbang:
        # turn into bang-bang control to reduce the peak steering value, but the mean heading value is not retained
            pos_flag = (yawrate>0)
            neg_flag = ~pos_flag
            mean_pos_steering = (yawrate*pos_flag).sum(-1)/((c0*pos_flag).sum(-1)+1e-6)
            mean_neg_steering = (yawrate*neg_flag).sum(-1)/((c0*neg_flag).sum(-1)+1e-6)
            mean_pos_steering = torch.clip(mean_pos_steering,min=-self.max_steer,max=self.max_steer)
            mean_neg_steering = torch.clip(mean_neg_steering,min=-self.max_steer,max=self.max_steer)
            bb_yawrate = (mean_pos_steering[...,None]*pos_flag+mean_neg_steering[...,None]*neg_flag)*c0
            bb_yawrate = torch.clip(bb_yawrate,min=-self.max_rvel/c0,max=self.max_rvel/c0)
            dh = torch.cumsum(bb_yawrate*self.dt,-1)
        else:
            yawrate = torch.clip(yawrate,min=-self.max_rvel/c0,max=self.max_rvel/c0)
            yawrate = torch.clip(yawrate,min=-self.max_steer*c0,max=self.max_steer*c0)
            dh = torch.cumsum(yawrate*self.dt,-1)
        heading = x0[...,3,None,None,None]+dh
        
        vx = vplan*torch.cos(heading)
        vy = vplan*torch.sin(heading)
        traj = torch.stack([x0[:,None,None,None,0]+vx.cumsum(-1)*self.dt,
                            x0[:,None,None,None,1]+vy.cumsum(-1)*self.dt,
                            vplan.repeat_interleave(M,2),
                            heading],-1)
        t = torch.arange(1,T+1,device=self.device)[None,None,None,:,None].repeat(bs,N,M,1,1)*self.dt
        xyvaqrt = torch.cat([traj[...,:3],acce[...,None].repeat_interleave(M,2),traj[...,3:],yawrate[...,None],t],-1)
        return xyvaqrt.reshape(bs,N*M,T,-1) # [B, N*M, T, 7]
        
    def sample_trajs(self,x0,T,bangbang=True):
        # velocity sample
        vplan = self.velocity_plan(x0,T)
        bs = x0.shape[0]
        dhf = self.dhf_grid
        dhm = self.dhm_grid
        Mf = dhf.shape[0]
        Mm = dhm.shape[0]
        dhm = dhm.repeat(Mf).unsqueeze(0).repeat_interleave(bs,0)
        dhf = dhf.repeat_interleave(Mm,0).unsqueeze(0).repeat_interleave(bs,0)+dhm
        return self.lateral_plan(x0,vplan,dhf,dhm,T,bangbang)
        
        
        
def test():
    sampler = ForwardSampler(acce_grid=[-4,-2,0,2,4],dhm_grid=torch.linspace(-0.7,0.7,9),dhf_grid=[-0.4,0,0.4],dt=0.1)
    x0 = torch.tensor([0,0,1.,0.],device="cuda").unsqueeze(0).repeat_interleave(3,0)
    T = 10
    # vel_grid = sampler.velocity_plan(x0,T)
    # dhf = torch.tensor([0.5,0,-0.5]).repeat(3).unsqueeze(0)
    # dhm = torch.tensor([0.2,0,-0.2]).repeat_interleave(3,0).unsqueeze(0)
    traj = sampler.sample_trajs(x0,T,bangbang=False)
    traj = traj[0].reshape(-1,T,7).cpu().numpy()
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    for i in range(traj.shape[0]):
        ax.plot(traj[i,:,0],traj[i,:,1])
    plt.show()
    
    
        
if __name__ == "__main__":
    test()
        
        
        