import torch
import torch.nn as nn
import numpy as np
import random

from filing_paths import path_model

import sys
from Extended_data import N_E, N_CV, N_T
# Number of Training Epochs
N_Epochs = 100
# Number of Samples in Batch
N_B = 10
# Learning Rate
learning_rate = 1e-3
# L2 Weight Regularization - Weight Decay
wd = 1e-6

sys.path.insert(1, path_model)
from model import f, h

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("using GPU!")
else:
    dev = torch.device("cpu")
    print("using CPU!")


def NNTrain(SysModel, Model, cv_input, cv_target, train_input, train_target, path_results, nclt=False, sequential_training=False, rnn=False, epochs=None, train_IC=None, CV_IC=None):

    N_E = train_input.size()[0]
    N_CV = cv_input.size()[0]

    Model.to(dev, non_blocking=True)


    # MSE LOSS Function
    loss_fn = nn.MSELoss(reduction='mean')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=wd)

    MSE_cv_linear_batch = torch.empty([N_CV]).to(dev, non_blocking=True)
    MSE_cv_linear_epoch = torch.empty([N_Epochs]).to(dev, non_blocking=True)
    MSE_cv_dB_epoch = torch.empty([N_Epochs]).to(dev, non_blocking=True)

    MSE_train_linear_batch = torch.empty([N_B]).to(dev, non_blocking=True)
    MSE_train_linear_epoch = torch.empty([N_Epochs]).to(dev, non_blocking=True)
    MSE_train_dB_epoch = torch.empty([N_Epochs]).to(dev, non_blocking=True)


    ##############
    ### Epochs ###
    ##############

    MSE_cv_dB_opt = 1000
    MSE_cv_idx_opt = 0

    if epochs is None:
        N = N_Epochs
    else:
        N = epochs

    for ti in range(0, N):

        #################################
        ### Validation Sequence Batch ###
        #################################

        # Cross Validation Mode
        Model.eval()

        for j in range(0, N_CV):
            Model.i = 0
            # Initialize next sequence
            if(sequential_training):
                if(nclt):
                    init_conditions = torch.reshape(cv_input[j,:,0], SysModel.m1x_0.shape)
                elif CV_IC is None:
                    init_conditions = torch.reshape(cv_target[j,:,0], SysModel.m1x_0.shape)
                else:
                    init_conditions = SysModel.m1x_0
            else:
                init_conditions = SysModel.m1x_0

            Model.InitSequence(init_conditions, SysModel.m2x_0, SysModel.T_test)

            y_cv = cv_input[j, :, :]

            x_Net_cv = torch.empty(SysModel.m, SysModel.T_test).to(dev, non_blocking=True)
            cv_target = cv_target.to(dev, non_blocking=True)

            for t in range(0, SysModel.T_test):
                x_Net_cv[:,t] = Model(y_cv[:,t])
            

            # Compute Training Loss
            if(nclt):
                if x_Net_cv.size()[0]==6:
                    mask = torch.tensor([True,False,False,True,False,False])
                else:
                    mask = torch.tensor([True,False,True,False])
                MSE_cv_linear_batch[j] = loss_fn(x_Net_cv[mask], cv_target[j, :, :]).item()
            else:
                MSE_cv_linear_batch[j] = loss_fn(x_Net_cv, cv_target[j, :, :]).item()

        # Average
        MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
        MSE_cv_dB_epoch[ti] = 10 * torch.log10(MSE_cv_linear_epoch[ti])

        if(MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt):

            MSE_cv_dB_opt = MSE_cv_dB_epoch[ti]
            MSE_cv_idx_opt = ti
            if(rnn):
                torch.save(Model, path_results+'best-model_rnn.pt')
            else:
                torch.save(Model, path_results+'best-model.pt')

        ###############################
        ### Training Sequence Batch ###
        ###############################

        # Training Mode
        Model.train()

        # Init Hidden State
        Model.init_hidden()

        Batch_Optimizing_LOSS_sum = 0

        for j in range(0, N_B):
            Model.i = 0
            n_e = random.randint(0, N_E - 1)

            y_training = train_input[n_e, :, :]

            if(sequential_training):
                if(nclt):
                    init_conditions = torch.reshape(cv_input[j,:,0], SysModel.m1x_0.shape)
                elif CV_IC is None:
                    init_conditions = torch.reshape(cv_target[j,:,0], SysModel.m1x_0.shape)
                else:
                    init_conditions = SysModel.m1x_0
            else:
                init_conditions = SysModel.m1x_0


            Model.InitSequence(init_conditions, SysModel.m2x_0, SysModel.T)
            

            x_Net_training = torch.empty(SysModel.m, SysModel.T).to(dev, non_blocking=True)
            train_target = train_target.to(dev, non_blocking=True)
            
            for t in range(0, SysModel.T):
                x_Net_training[:,t] = Model(y_training[:,t])

            # Compute Training Loss
            #LOSS = loss_fn(x_Net_training, train_target[n_e, :, :])
            if(nclt):
                if x_Net_training.size()[0]==6:
                    mask = torch.tensor([True,False,False,True,False,False])
                else:
                    mask = torch.tensor([True,False,True,False])
                LOSS = loss_fn(x_Net_training[mask], train_target[n_e, :, :])
            else:
                LOSS = loss_fn(x_Net_training, train_target[n_e, :, :])

            MSE_train_linear_batch[j] = LOSS.item()

            Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS
            #print(x_Net_training)



        # Average
        MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
        MSE_train_dB_epoch[ti] = 10 * torch.log10(MSE_train_linear_epoch[ti])

        ##################
        ### Optimizing ###
        ##################

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()


        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / N_B
        Batch_Optimizing_LOSS_mean.backward(retain_graph=True)

        #torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=2.0, norm_type=2)


        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        ########################
        ### Training Summary ###
        ########################
        print(ti, "MSE Training :", MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", MSE_cv_dB_epoch[ti], "[dB]")

        if (ti > 1):
            d_train = MSE_train_dB_epoch[ti] - MSE_train_dB_epoch[ti - 1]
            d_cv    = MSE_cv_dB_epoch[ti] - MSE_cv_dB_epoch[ti - 1]
            print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

        print("Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]")

    return [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch]


