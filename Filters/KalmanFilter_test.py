import torch
import torch.nn as nn
import time
from Filters.Linear_KF import KalmanFilter

def KFTest(args, SysModel, test_input, test_target, allStates=True,\
     randomInit = False, test_init=None, test_lengthMask=None):

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.zeros(args.N_T)
    # allocate memory for KF output
    KF_out = torch.zeros(args.N_T, SysModel.m, args.T_test)
    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    start = time.time()

    KF = KalmanFilter(SysModel, args)
    # Init and Forward Computation 
    if(randomInit):
        KF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))        
    else:
        KF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(args.N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))           
    KF.GenerateBatch(test_input)
    
    end = time.time()
    t = end - start
    KF_out = KF.x
    # MSE loss
    for j in range(args.N_T):# cannot use batch due to different length and std computation   
        if(allStates):
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j,:,:], test_target[j,:,:]).item()
        else: # mask on state
            if args.randomLength:
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_KF_linear_arr[j] = loss_fn(KF.x[j,loc,:], test_target[j,loc,:]).item()

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)

    # Confidence interval
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out]



