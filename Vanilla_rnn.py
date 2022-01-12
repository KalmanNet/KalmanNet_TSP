"""# **Class: Vanilla RNN**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

in_mult = 3
out_mult = 2
nGRU = 3

class Vanilla_RNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################

    def Build(self, SysModel, fully_agnostic = True):
        self.fully_agnostic = fully_agnostic

        # Set State Evolution Function
        self.f = SysModel.f
        self.m = SysModel.m

        # Set Observation Function
        self.h = SysModel.h
        self.n = SysModel.n

        self.InitSequence(SysModel.m1x_0, SysModel.T)

        # input dim for GRU
        input_dim_RNN = (self.m + self.n) * in_mult
        # Hidden Dimension for GRU
        hidden_dim_RNN = ((self.n * self.n) + (self.m * self.m)) * out_mult

        self.InitRNN(input_dim_RNN, hidden_dim_RNN)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitRNN(self, input_dim_RNN, hidden_dim_RNN):

        self.seq_len_input = 1
        self.batch_size = 1
        self.n_layers = nGRU
        self.hidden_dim = hidden_dim_RNN
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim)

        # GRUs
        self.rnn_GRU = nn.GRU(input_dim_RNN, self.hidden_dim, self.n_layers)
        
        # Fully connected 1
        self.d_input_FC1 = self.m + self.n
        self.d_output_FC1 = input_dim_RNN
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.hidden_dim
        self.d_output_FC2 = self.m 
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_output_FC2),
                nn.ReLU())
        

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.xhat_previous)


    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        self.T = T
        self.x_out = torch.empty(self.m, T).to(dev, non_blocking=True)

        self.xhat = M1_0.to(dev, non_blocking=True)
        self.xhat_previous = self.xhat.to(dev, non_blocking=True)

    def step_est(self, y):

        xhat = self.xhat_step(self.xhat_previous, y)

        # Reshape Kalman Gain to a Matrix
        self.xhat = torch.reshape(xhat, (self.m, 1))

    ########################
    ### Vanilla RNN Step ###
    ########################
    def RNN_step(self, y):

        if self.fully_agnostic:
            self.step_est(y)

            self.xhat_previous = self.xhat

            return torch.squeeze(self.xhat)
        else:
            # Compute Priors
            self.step_prior()

            self.step_est(y)

            self.xhat = self.m1x_prior + self.xhat

            self.xhat_previous = self.xhat

            return torch.squeeze(self.xhat)


    ########################
    ### Kalman Gain Step ###
    ########################
    def xhat_step(self, m1x_posterior_previous, y):

        def expand_dim(x):
            x = torch.squeeze(x)
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded
        
        m1x_posterior_previous = expand_dim(m1x_posterior_previous)
        y = expand_dim(y)

        ####################
        ### Forward Flow ###
        ####################       
        # FC 1
        in_FC1 = torch.cat((m1x_posterior_previous, y), 2)
        out_FC1 = self.FC1(in_FC1)
       
        # GRU        
        GRU_out, self.hn = self.rnn_GRU(expand_dim(out_FC1), self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        # FC 2
        in_FC2 = GRU_out_reshape 
        out_FC2 = self.FC2(in_FC2)

        return out_FC2

    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        y = y.to(dev, non_blocking=True)

        self.x_out = self.RNN_step(y)

        return self.x_out

     #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data



