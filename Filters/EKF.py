"""# **Class: Extended Kalman Filter**
Theoretical Non Linear Kalman
"""
import torch

from Simulations.Lorenz_Atractor.parameters import getJacobian

class ExtendedKalmanFilter:

    def __init__(self, SystemModel):
        self.f = SystemModel.f
        self.m = SystemModel.m

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q

        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test
   
    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(self.m1x_posterior,self.f), getJacobian(self.m1x_prior, self.h))
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)
        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

        #Save KalmanGain
        self.KG_array[:,:,:,self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    #########################

    def UpdateJacobians(self, F, H):
        self.batched_F = F
        self.batched_F_T = torch.transpose(F,1,2)
        self.batched_H = H
        self.batched_H_T = torch.transpose(H,1,2)
    
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

            self.m1x_0_batch = m1x_0_batch # [batch_size, m, 1]
            self.m2x_0_batch = m2x_0_batch # [batch_size, m, m]

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, y):
        """
        input y: batch of observations [batch_size, n, T]
        """
        self.batch_size = y.shape[0] # batch size
        T = y.shape[2] # sequence length (maximum length if randomLength=True)

        # Pre allocate KG array
        self.KG_array = torch.zeros([self.batch_size,self.m,self.n,T])
        self.i = 0 # Index for KG_array alocation

        # Allocate Array for 1st and 2nd order moments (use zero padding)
        self.x = torch.zeros(self.batch_size, self.m, T)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T)
            
        # Set 1st and 2nd order moments for t=0
        self.m1x_posterior = self.m1x_0_batch
        self.m2x_posterior = self.m2x_0_batch

        # Generate in a batched manner
        for t in range(0, T):
            yt = torch.unsqueeze(y[:, :, t],2)
            xt,sigmat = self.Update(yt)
            self.x[:, :, t] = torch.squeeze(xt,2)
            self.sigma[:, :, :, t] = sigmat