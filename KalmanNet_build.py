from KalmanNet_nn import KalmanNetNN
import torch

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

def NNBuild(SysModel):

    Model = KalmanNetNN()

    Model.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n, infoString = "partialInfo")
    Model.InitSequence(SysModel.m1x_0, SysModel.m2x_0, SysModel.T)

    # Number of neurons in the 1st hidden layer
    #H1_KNet = (SysModel.m + SysModel.n) * (10) * 8

    # Number of neurons in the 2nd hidden layer
    #H2_KNet = (SysModel.m * SysModel.n) * 1 * (4)

    Model.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S)

    return Model

