"""
This file contains the parameters for the simulations with linear kinematic model
* Constant Acceleration Model (CA)
    # full state P, V, A
    # only postion P
* Constant Velocity Model (CV)
"""

import torch

m = 3 # dim of state for CA model
m_cv = 2 # dim of state for CV model

delta_t_gen =  1e-2

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################
F_gen = torch.tensor([[1, delta_t_gen,0.5*delta_t_gen**2],
                  [0,       1,       delta_t_gen],
                  [0,       0,         1]]).float()

F_CV = torch.tensor([[1, delta_t_gen],
                     [0,           1]]).float()              

# Full observation
H_identity = torch.eye(3)
# Observe only the postion
H_onlyPos = torch.tensor([[1, 0, 0]]).float()

###############################################
### process noise Q and observation noise R ###
###############################################
# Noise Parameters
r2 = torch.tensor([1]).float()
q2 = torch.tensor([1]).float()

Q_gen = q2 * torch.tensor([[1/20*delta_t_gen**5, 1/8*delta_t_gen**4,1/6*delta_t_gen**3],
                           [ 1/8*delta_t_gen**4, 1/3*delta_t_gen**3,1/2*delta_t_gen**2],
                           [ 1/6*delta_t_gen**3, 1/2*delta_t_gen**2,       delta_t_gen]]).float()

Q_CV = q2 * torch.tensor([[1/3*delta_t_gen**3, 1/2*delta_t_gen**2],
                          [1/2*delta_t_gen**2,        delta_t_gen]]).float()  

R_3 = r2 * torch.eye(3)
R_2 = r2 * torch.eye(2)

R_onlyPos = r2