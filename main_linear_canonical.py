import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel
from Simulations.utils import DataGen
import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure,\
   m, m1_0

from Filters.KalmanFilter_test import KFTest

from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline_EKF import Pipeline_EKF 

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'KNet/'

####################
### Design Model ###
####################
args = config.general_settings()

### dataset parameters ##################################################
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
# init condition
args.randomInit_train = False
args.randomInit_cv = False
args.randomInit_test = False
if args.randomInit_train or args.randomInit_cv or args.randomInit_test:
   # you can modify initial variance
   args.variance = 1
   args.distribution = 'normal' # 'uniform' or 'normal'
   m2_0 = args.variance * torch.eye(m)
else: 
   # deterministic initial condition
   m2_0 = 0 * torch.eye(m) 
# sequence length
args.T = 100
args.T_test = 100
args.randomLength = False
if args.randomLength:# you can modify T_max and T_min 
   args.T_max = 1000
   args.T_min = 100
   # set T and T_test to T_max for convenience of batch calculation
   args.T = args.T_max 
   args.T_test = args.T_max
else:
   train_lengthMask = None
   cv_lengthMask = None
   test_lengthMask = None
# noise
r2 = torch.tensor([1])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

### training parameters ##################################################
args.use_cuda = True # use GPU or not
args.n_steps = 4000
args.n_batch = 30
args.lr = 1e-4
args.wd = 1e-3

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

### True model ##################################################
Q = q2 * Q_structure
R = r2 * R_structure
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/data' + '/'
dataFileName = '2x2_rq020_T100.pt'
print("Start Data Gen")
DataGen(args, sys_model, dataFolderName + dataFileName)
print("Data Load")
if args.randomLength:
   [train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask] = torch.load(dataFolderName + dataFileName, map_location=device)
else:
   [train_input, train_target, cv_input, cv_target, test_input, test_target,_,_,_] = torch.load(dataFolderName + dataFileName, map_location=device)

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())

########################################
### Evaluate Observation Noise Floor ###
########################################
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(args.N_T)# MSE [Linear]  
for i in range(args.N_T):
   MSE_obs_linear_arr[i] = loss_obs(test_input[i], test_target[i]).item()   
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor - STD:", obs_std_dB, "[dB]")

##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter True")
if args.randomInit_test:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model, test_input, test_target, randomInit = True, test_init=test_init, test_lengthMask=test_lengthMask)
else: 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model, test_input, test_target, test_lengthMask=test_lengthMask)


##########################
### KalmanNet Pipeline ###
##########################

### KalmanNet with full info ##########################################################################################
# Build Neural Network
print("KalmanNet with full model info")
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
## Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(args)
if (args.randomInit_train or args.randomInit_cv or args.randomInit_test):
   if args.randomLength:
      ## Train Neural Network
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit = True, cv_init=cv_init,train_init=train_init,train_lengthMask=train_lengthMask,cv_lengthMask=cv_lengthMask)
      ## Test Neural Network
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,randomInit=True,test_init=test_init,test_lengthMask=test_lengthMask)
   else:    
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit = True, cv_init=cv_init,train_init=train_init)
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,randomInit=True,test_init=test_init)
else:
   if args.randomLength:
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,train_lengthMask=train_lengthMask,cv_lengthMask=cv_lengthMask)
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,test_lengthMask=test_lengthMask)
   else:
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
KalmanNet_Pipeline.save()