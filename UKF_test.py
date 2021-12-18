import torch.nn as nn
import torch
import time

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints 


def UKFTest(SysModel, test_input, test_target, modelKnowledge='full', allStates=True, init_cond=None):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    
    # MSE [Linear]
    MSE_UKF_linear_arr = torch.empty(N_T)
    # points = JulierSigmaPoints(n=SysModel.m)
    points = MerweScaledSigmaPoints(SysModel.m, alpha=.1, beta=2., kappa=-1)

    def fx(x, dt):
        return SysModel.f(torch.from_numpy(x).float()).numpy()

    def hx(x):
        return SysModel.h(torch.from_numpy(x).float()).numpy()

    UKF = UnscentedKalmanFilter(dim_x=SysModel.m, dim_z=SysModel.n, dt=SysModel.delta_t, fx=fx, hx=hx,points=points)
    UKF.x = SysModel.m1x_0.numpy() # initial state
    UKF.P = (SysModel.m2x_0 + 1e-5*torch.eye(SysModel.m)).numpy() # initial uncertainty
    UKF.R = SysModel.R.numpy()
    UKF.Q = SysModel.Q.numpy()
 
    UKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test]) 

    start = time.time()
    for j in range(0, N_T):
        if init_cond is not None:
            UKF.x = torch.unsqueeze(init_cond[j, :], 1).numpy()
        
        for z in range(0, SysModel.T_test):
            UKF.predict()
            UKF.update(test_input[j,:,z].numpy())       
            UKF_out[j,:,z] = torch.from_numpy(UKF.x)

        if allStates:
            MSE_UKF_linear_arr[j] = loss_fn(UKF_out[j,:,:], test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True, False, True, False])
            MSE_UKF_linear_arr[j] = loss_fn(UKF_out[j,:,:][loc, :], test_target[j, :, :]).item()

    end = time.time()
    t = end - start

    MSE_UKF_linear_avg = torch.mean(MSE_UKF_linear_arr)
    MSE_UKF_dB_avg = 10 * torch.log10(MSE_UKF_linear_avg)
    # Standard deviation
    MSE_UKF_dB_std = torch.std(MSE_UKF_linear_arr, unbiased=True)
    MSE_UKF_dB_std = 10 * torch.log10(MSE_UKF_dB_std)

    print("UKF - MSE LOSS:", MSE_UKF_dB_avg, "[dB]")
    print("UKF - MSE STD:", MSE_UKF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out]
