"""# **Class: System Model for Non-linear Cases**

1 Store system model parameters: 
    state transition function f, 
    observation function h, 
    process noise Q, 
    observation noise R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test,
    state dimension m,
    observation dimension n, etc.

2 Generate datasets for non-linear cases
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class SystemModel:

    def __init__(self, f, Q, h, R, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.m = m
        self.Q = Q
        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = n
        self.R = R
        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################   
            if torch.equal(Q_gen,torch.zeros(self.m,self.m)):# No noise
                 xt = self.f(self.x_prev)   
            elif self.m == 1: # 1 dim noise
                xt = self.f(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:            
                xt = self.f(self.x_prev)
                mean = torch.zeros([self.m])              
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            yt = self.h(xt)
            # Observation Noise         
            if self.n == 1: # 1 dim noise
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:  
                mean = torch.zeros([self.n])            
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:], yt.size())       
                # Additive Observation Noise
                yt = torch.add(yt,er)
            
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt,1)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt,1)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, args, size, T, randomInit=False):
        if(randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            if args.distribution == 'uniform':
                ### if Uniform Distribution for random init
                for i in range(size):           
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                    self.m1x_0_rand[i,:,0:1] = initConditions.view(self.m,1)     
            
            elif args.distribution == 'normal':
                ### if Normal Distribution for random init
                for i in range(size):
                    distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                    initConditions = distrib.rsample().view(self.m,1)
                    self.m1x_0_rand[i,:,0:1] = initConditions
            else:
                raise ValueError('args.distribution not supported!')
            
            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)### for sequence generation
        else: # fixed init
            initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)
            self.Init_batched_sequence(initConditions, self.m2x_0)### for sequence generation
    
        if(args.randomLength):
            # Allocate Array for Input and Target (use zero padding)
            self.Input = torch.zeros(size, self.n, args.T_max)
            self.Target = torch.zeros(size, self.m, args.T_max)
            self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool)# init with all false
            # Init Sequence Lengths
            T_tensor = torch.round((args.T_max-args.T_min)*torch.rand(size)).int()+args.T_min # Uniform distribution [100,1000]
            for i in range(0, size):
                # Generate Sequence
                self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
                # Training sequence input
                self.Input[i, :, 0:T_tensor[i].item()] = self.y             
                # Training sequence output
                self.Target[i, :, 0:T_tensor[i].item()] = self.x
                # Mask for sequence length
                self.lengthMask[i, 0:T_tensor[i].item()] = True

        else:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.m, T)

            # Set x0 to be x previous
            self.x_prev = self.m1x_0_batch
            xt = self.x_prev

            # Generate in a batched manner
            for t in range(0, T):
                ########################
                #### State Evolution ###
                ########################   
                if torch.equal(self.Q,torch.zeros(self.m,self.m)):# No noise
                    xt = self.f(self.x_prev)
                elif self.m == 1: # 1 dim noise
                    xt = self.f(self.x_prev)
                    eq = torch.normal(mean=torch.zeros(size), std=self.Q).view(size,1,1)
                    # Additive Process Noise
                    xt = torch.add(xt,eq)
                else:            
                    xt = self.f(self.x_prev)
                    mean = torch.zeros([size, self.m])              
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
                    eq = distrib.rsample().view(size,self.m,1)
                    # Additive Process Noise
                    xt = torch.add(xt,eq)

                ################
                ### Emission ###
                ################
                # Observation Noise
                if torch.equal(self.R,torch.zeros(self.n,self.n)):# No noise
                    yt = self.h(xt)
                elif self.n == 1: # 1 dim noise
                    yt = self.h(xt)
                    er = torch.normal(mean=torch.zeros(size), std=self.R).view(size,1,1)
                    # Additive Observation Noise
                    yt = torch.add(yt,er)
                else:  
                    yt =  self.h(xt)
                    mean = torch.zeros([size,self.n])            
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
                    er = distrib.rsample().view(size,self.n,1)          
                    # Additive Observation Noise
                    yt = torch.add(yt,er)

                ########################
                ### Squeeze to Array ###
                ########################

                # Save Current State to Trajectory Array
                self.Target[:, :, t] = torch.squeeze(xt,2)

                # Save Current Observation to Trajectory Array
                self.Input[:, :, t] = torch.squeeze(yt,2)

                ################################
                ### Save Current to Previous ###
                ################################
                self.x_prev = xt
