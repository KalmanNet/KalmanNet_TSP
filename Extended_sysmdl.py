import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import delta_t, delta_t_gen, variance

class SystemModel:

    def __init__(self, f, q, h, r, T, T_test, m, n, modelname):

        ####################
        ### Motion Model ###
        ####################
        self.modelname = modelname

        self.f = f
        self.m = m

        self.q = q

        self.delta_t = delta_t
        if self.modelname == 'pendulum':
            self.Q = q * q * torch.tensor([[(delta_t**3)/3, (delta_t**2)/2],
                                           [(delta_t**2)/2, delta_t]])
        elif self.modelname == 'pendulum_gen':
            self.Q = q * q * torch.tensor([[(delta_t_gen**3)/3, (delta_t_gen**2)/2],
                                           [(delta_t_gen**2)/2, delta_t_gen]])
        else:
            self.Q = q * q * torch.eye(self.m)

        

        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = n

        self.r = r
        self.R = r * r * torch.eye(self.n)

        #Assign T and T_test
        self.T = T
        self.T_test = T_test

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = torch.squeeze(m1x_0)
        self.m2x_0 = torch.squeeze(m2x_0)


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0

        # Generate Sequence Iteratively
        for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################
            # Process Noise
            if self.q == 0:
                xt = self.f(self.x_prev)              
            else:
                xt = self.f(self.x_prev)
                mean = torch.zeros([self.m])
                if self.modelname == "pendulum":
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                    eq = distrib.rsample()
                else:
                    eq = torch.normal(mean, self.q)
                         
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            yt = self.h(xt)

            # Observation Noise
            mean = torch.zeros([self.n])
            er = torch.normal(mean, self.r)
            # er = np.random.multivariate_normal(mean, R_gen, 1)
            # er = torch.transpose(torch.tensor(er), 0, 1)

            # Additive Observation Noise
            yt = torch.add(yt,er)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, randomInit=False):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        initConditions = self.m1x_0 

        ### Generate Examples
        for i in range(0, size):
            # Generate Sequence
            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                initConditions = torch.rand_like(self.m1x_0) * variance
            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x

