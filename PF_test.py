import pyparticleest.models.nlg
import pyparticleest.simulator as simulator
import pyparticleest.utils.kalman as kalman
import time
import torch
import torch.nn as nn
import numpy as np

class Model(pyparticleest.models.nlg.NonlinearGaussianInitialGaussian):

    def __init__(self, SystemModel, x_0=None):
        if x_0 == None:
            x0 = SystemModel.m1x_0
        else:
            x0 = x_0
        P0 = SystemModel.m2x_0
        Q = SystemModel.Q.numpy()
        R = SystemModel.R.numpy()
        super(Model, self).__init__(x0=x0, Px0=P0, Q=Q, R=R)
        self.f = SystemModel.f
        self.n = SystemModel.n
        self.g = lambda x: torch.squeeze(SystemModel.h(x))
        self.m = SystemModel.m


    def calc_f(self, particles, u, t):
        N_p = particles.shape[0]
        particles_f = np.empty((N_p, self.n))
        for k in range(N_p):
            particles_f[k,:] = self.f(torch.tensor(particles[k,:]))
        return particles_f

    def calc_g(self, particles, t):
        N_p = particles.shape[0]
        particles_g = np.empty((N_p, self.m))
        for k in range(N_p):
            particles_g[k,:] = self.g(torch.tensor(particles[k,:]))
        return particles_g

def PFTest(SysModel, test_input, test_target, n_part=100):
    
    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_PF_linear_arr = torch.empty(N_T)

    PF_out = torch.empty([N_T, SysModel.m, SysModel.T_test])

    start = time.time()

    for j in range(N_T):
        model = Model(SysModel, test_target[j, :, 0])
        y_in = test_input[j, :, :].T.numpy().squeeze()
        sim = simulator.Simulator(model, u=None, y=y_in)
        sim.simulate(n_part, 0)
        PF_out[j, :, :] = torch.from_numpy(sim.get_filtered_mean()[1:,].T).float()


    for j in range(N_T):
        MSE_PF_linear_arr[j] = loss_fn(torch.tensor(PF_out[j, :, :]), test_target[j, :, :])

    end = time.time()
    t = end - start

    MSE_PF_linear_avg = torch.mean(MSE_PF_linear_arr)
    MSE_PF_dB_avg = 10 * torch.log10(MSE_PF_linear_avg)
    # Standard deviation
    MSE_PF_dB_std = torch.std(MSE_PF_linear_arr, unbiased=True)
    MSE_PF_dB_std = 10 * torch.log10(MSE_PF_dB_std)

    print("PF - MSE LOSS:", MSE_PF_dB_avg, "[dB]")
    print("PF - MSE STD:", MSE_PF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out]


