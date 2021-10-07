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
m = 4
n = 4
T = 20
T_test = 20

m1x_0 = torch.zeros(m, 1)
m2x_0 = torch.tensor([[0,0,0,0],
                      [0,0,0,0],
                      [0,0,0,0],
                      [0,0,0,0]]).float()

# Time step span
delta_t = 1

##########################################
### Dynamics For No Air Friction Movement ###
##########################################
# Taylor approximation
F_dim = torch.tensor([[1.0, delta_t],
                      [0,1.0]])


# Model Transformations
F_design = torch.block_diag(F_dim, F_dim)

H_design = torch.eye(n) 
'''
H_design = torch.tensor([[0,1.,0,0],
                          [0,0,0,1.]])
'''
# Noise Parameters
lambda_q_mod = 1
lambda_r_mod = 1

# Noise matrix per dimension
Q_dim = torch.diagflat(torch.tensor([delta_t, delta_t]))

# Noise Matrices
Q = (lambda_q_mod**2) * torch.block_diag(Q_dim, Q_dim)


R = torch.tensor([[700,0,0,0],
                  [0,0.01,0,0],
                  [0,0,700,0],
                  [0,0,0,0.01]]).float()
'''
R = torch.eye(n) * (sigma_r**2)
'''