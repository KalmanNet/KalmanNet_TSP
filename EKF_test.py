import torch.nn as nn
import torch
import time
from EKF import ExtendedKalmanFilter


def EKFTest(SysModel, test_input, test_target, modelKnowledge = 'full', allStates=True):

    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    
    # MSE [Linear]
    MSE_EKF_linear_arr = torch.empty(N_T)

    EKF = ExtendedKalmanFilter(SysModel, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KG_array = torch.zeros_like(EKF.KG_array)
    EKF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])
    start = time.time()
    for j in range(0, N_T):
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        if(allStates):
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        else:
            loc = torch.tensor([True,False,True,False])
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc,:], test_target[j, :, :]).item()
        KG_array = torch.add(EKF.KG_array, KG_array) 
        EKF_out[j,:,:] = EKF.x
    end = time.time()
    t = end - start
    # Average KG_array over Test Examples
    KG_array /= N_T

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)
    
    print("EKF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("EKF - MSE STD:", MSE_EKF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]



