import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class SystemModel:

    def __init__(self, F, q, H, r, T, T_test, outlier_p=0,rayleigh_sigma=10000):

        self.outlier_p = outlier_p
        self.rayleigh_sigma = rayleigh_sigma
        ####################
        ### Motion Model ###
        ####################       
        self.F = F
        self.m = self.F.size()[0]

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]

        self.r = r
        self.R = r * r * torch.eye(self.n)

        #Assign T and T_test
        self.T = T
        self.T_test = T_test

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0


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
        
        # Outliers
        if self.outlier_p > 0:
            b_matrix = torch.bernoulli(self.outlier_p *torch.ones(T))

        # Generate Sequence Iteratively
        for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################            
            # Process Noise
            if self.q == 0:
                xt = self.F.matmul(self.x_prev)            
            else:
                xt = self.F.matmul(self.x_prev)
                mean = torch.zeros([self.m])              
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                # eq = torch.normal(mean, self.q)
                eq = torch.reshape(eq[:],[self.m,1])
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            # Observation Noise
            if self.r == 0:
                yt = self.H.matmul(xt)           
            else:
                yt = self.H.matmul(xt)
                mean = torch.zeros([self.n])            
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:],[self.n,1])
                # mean = torch.zeros([self.n,1])
                # er = torch.normal(mean, self.r)
                
                # Additive Observation Noise
                yt = torch.add(yt,er)
            
            # Outliers
            if self.outlier_p > 0:
                if b_matrix[t] != 0:
                    btdt = self.rayleigh_sigma*torch.sqrt(-2*torch.log(torch.rand(self.n,1)))
                    yt = torch.add(yt,btdt)

            


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

    def GenerateBatch(self, size, T, randomInit=False, seqInit=False, T_test=0):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        ### Generate Examples
        initConditions = self.m1x_0

        for i in range(0, size):
            # Generate Sequence

            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if(seqInit):
                initConditions = self.x_prev
                if((i*T % T_test)==0):
                    initConditions = torch.zeros_like(self.m1x_0)

            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x

