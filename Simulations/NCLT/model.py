# import autograd.numpy as np
from autograd import grad, jacobian
import torch
from torch import autograd

from parameters import F_design, H_design

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

def f(x):
    return torch.matmul(F_design, x)

def h(x):
    return torch.matmul(H_design, x)

def fInacc(x):
    return torch.matmul(F_mod, x)

def hInacc(x):
    return torch.matmul(H_mod, x)

def getJacobian(x, a):
    
    try:
        if(x.size()[1] == 1):
            y = torch.reshape((x.T),[x.size()[0]])
    except:
        y = x

    if(a == 'ObsAcc'):
        g = h
    elif(a == 'ModAcc'):
        g = f
    elif(a == 'ObsInacc'):
        g = hInacc
    elif(a == 'ModInacc'):
        g = fInacc

    return autograd.functional.jacobian(g, y)
