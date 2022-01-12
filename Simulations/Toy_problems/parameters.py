import torch
import math

#########################
### Design Parameters ###
#########################
m = 2
n = 2

m1x_0 = torch.ones(m, 1) * 0.1
m2x_0 = torch.zeros(m,m)

T = 100
T_test = 100
#######################
### True Parameters ###
#######################
alpha_mot = 0.9
beta_mot = 1.1
phi_mot = math.pi/10
a_mot = 1
alpha_obs = 1
beta_obs = 1
a_obs = 0

# Noise Parameters
sigma_q = 0.1
sigma_r = 0.1

# Noise Matrices
Q = (sigma_q**2) * torch.eye(m)
R = (sigma_r**2) * torch.eye(m)

########################
### Model Parameters ###
########################
alpha_mot_mod = 1
beta_mot_mod = 1
phi_mot_mod = 0
a_mot_mod = 1
beta_obs_mod = 1
a_obs_mod = 0

# Noise Parameters
lambda_q_mod = 0.7
lambda_r_mod = 0.1

# Noise Matrices
Q_mod = (lambda_q_mod**2) * torch.eye(m)
R_mod = (lambda_r_mod**2) * torch.eye(m)