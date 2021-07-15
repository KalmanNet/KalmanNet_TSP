import torch.nn as nn
import torch
import time

from filterpy.kalman import UnscentedKalmanFilter 

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n,delta_t_gen,delta_t
from model import f, h, fInacc, hInacc, fRotate


def UKFTest(SysModel, test_input, test_target, modelKnowledge='full', allStates=True, init_cond=None):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    
    # MSE [Linear]
    MSE_UKF_linear_arr = torch.empty(N_T)

    UKF = UnscentedKalmanFilter(dim_x=SysModel.m, dim_z=SysModel.n, dt=delta_t, fx=SysModel.f, hx=SysModel.h)
    UKF.x = SysModel.m1x_0 # initial state
    UKF.P = SysModel.m2x_0 # initial uncertainty
    UKF.R = SysModel.R 
    UKF.Q = SysModel.Q

    UKF_out = torch.empty([N_T, SysModel.m, SysModel.T])

    start = time.time()
    for j in range(0, N_T):
        if init_cond is not None:
            UKF.x = torch.unsqueeze(init_cond[j, :], 1)
        
        for z in test_input:
            UKF.predict()
            UKF.update(z)

        if allStates:
            MSE_UKF_linear_arr[j] = loss_fn(UKF.x, test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True, False, True, False])
            MSE_UKF_linear_arr[j] = loss_fn(UKF.x[loc, :], test_target[j, :, :]).item()
        UKF_out[j, :, :] = UKF.x
    end = time.time()
    t = end - start

    MSE_UKF_linear_avg = torch.mean(MSE_UKF_linear_arr)
    MSE_UKF_dB_avg = 10 * torch.log10(MSE_UKF_linear_avg)

    return [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out, t]
