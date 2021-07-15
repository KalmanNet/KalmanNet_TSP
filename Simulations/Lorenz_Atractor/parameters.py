import torch
import math

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

#########################
### Design Parameters ###
#########################
m = 3
n = 3
variance = 0
m1x_0 = torch.ones(m, 1) 
m1x_0_design_test = torch.ones(m, 1)
m2x_0 = 0 * 0 * torch.eye(m)

#################################################
### Generative Parameters For Lorenz Atractor ###
#################################################

# Auxiliar MultiDimensional Tensor B and C (they make A --> Differential equation matrix)
B = torch.tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(m,m), torch.zeros(m,m)]).float()
C = torch.tensor([[-10, 10,    0],
                  [ 28, -1,    0],
                  [  0,  0, -8/3]]).float()

delta_t_gen =  1e-5
delta_t = 0.02
delta_t_test = 0.01
J = 5

# Decimation ratio
ratio = delta_t_gen/delta_t_test

# Length of Time Series Sequence
# T = math.ceil(3000 / ratio)
# T_test = math.ceil(6e6 * ratio)
T = 20
T_test = 20

H_design = torch.eye(3)

## Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi/180)
yaw = yaw_deg * (math.pi/180)
pitch = pitch_deg * (math.pi/180)

RX = torch.tensor([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
RY = torch.tensor([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = torch.tensor([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

RotMatrix = torch.mm(torch.mm(RZ, RY), RX)
H_mod = torch.mm(RotMatrix,H_design)


H_design_inv = torch.inverse(H_design)

# Noise Parameters
r_dB = 0
lambda_r = math.sqrt(10**(-r_dB/10))
nx = 0
lambda_q = lambda_r * nx

# Noise Matrices
Q_non_diag = False
R_non_diag = False

Q = (lambda_q**2) * torch.eye(m)

if(Q_non_diag):
    q_d = lambda_q**2
    q_nd = (lambda_q **2)/2
    Q = torch.tensor([[q_d, q_nd, q_nd],[q_nd, q_d, q_nd],[q_nd, q_nd, q_d]])

R = (lambda_r**2) * torch.eye(n)

if(R_non_diag):
    r_d = lambda_r**2
    r_nd = (lambda_r **2)/2
    R = torch.tensor([[r_d, r_nd, r_nd],[r_nd, r_d, r_nd],[r_nd, r_nd, r_d]])

#########################
### Model Parameters ####
#########################

m1x_0_mod = m1x_0
m1x_0_mod_test = m1x_0_design_test
m2x_0_mod = 0 * 0 * torch.eye(m)

# Sampling time step
delta_t_mod = delta_t

# Length of Time Series Sequence
T_mod = math.ceil(T * ratio)
T_test_mod = math.ceil(T_test * ratio)

##############################################
#### Model Parameters For Lorenz Atractor ####
##############################################

# Auxiliar MultiDimensional Tensor B and C (they make A)
B_mod = torch.tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(m,m), torch.zeros(m,m)])
C_mod = torch.tensor([[-10, 10,    0],
                      [ 28, -1,    0],
                      [  0,  0, -8/3]])

J_mod = 2

# H_mod = torch.eye(n)
#H_mod = H_design
H_mod_inv = torch.inverse(H_mod)

# Noise Parameters
lambda_q_mod = 0.8
lambda_r_mod = 1

# Noise Matrices
Q_mod = (lambda_q_mod**2) * torch.eye(m)
R_mod = (lambda_r_mod**2) * torch.eye(n)