"""# **Class: Kalman Filter**
Theoretical Linear Kalman Filter
batched version
"""
import torch

class KalmanFilter:

    def __init__(self, SystemModel, args):
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.F = SystemModel.F
        self.m = SystemModel.m
        self.Q = SystemModel.Q.to(self.device)

        self.H = SystemModel.H
        self.n = SystemModel.n
        self.R = SystemModel.R.to(self.device)

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test
   
    # Predict

    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.bmm(self.batched_F, self.m1x_posterior).to(self.device)

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.Q

        # Predict the 1-st moment of y
        self.m1y = torch.bmm(self.batched_H, self.m1x_prior)

        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.R

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
               
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))

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

        return self.m1x_posterior,self.m2x_posterior

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
        y = y.to(self.device)
        self.batch_size = y.shape[0] # batch size
        T = y.shape[2] # sequence length (maximum length if randomLength=True)

        # Batched F and H
        self.batched_F = self.F.view(1,self.m,self.m).expand(self.batch_size,-1,-1).to(self.device)
        self.batched_F_T = torch.transpose(self.batched_F, 1, 2).to(self.device)
        self.batched_H = self.H.view(1,self.n,self.m).expand(self.batch_size,-1,-1).to(self.device)
        self.batched_H_T = torch.transpose(self.batched_H, 1, 2).to(self.device)

        # Allocate Array for 1st and 2nd order moments (use zero padding)
        self.x = torch.zeros(self.batch_size, self.m, T).to(self.device)
        self.sigma = torch.zeros(self.batch_size, self.m, self.m, T).to(self.device)
            
        # Set 1st and 2nd order moments for t=0
        self.m1x_posterior = self.m1x_0_batch.to(self.device)
        self.m2x_posterior = self.m2x_0_batch.to(self.device)

        # Generate in a batched manner
        for t in range(0, T):
            yt = torch.unsqueeze(y[:, :, t],2)
            xt,sigmat = self.Update(yt)
            self.x[:, :, t] = torch.squeeze(xt,2)
            self.sigma[:, :, :, t] = sigmat
