import torch
from datetime import datetime

from Filters.EKF_test import EKFTest

from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen,Short_Traj_Split
import Simulations.config as config
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,\
f, h, h_nonlinear, Q_structure, R_structure

from Pipelines.Pipeline_EKF import Pipeline_EKF

from KNet.KalmanNet_nn import KalmanNetNN

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

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 20
args.T_test = 20
### settings for KalmanNet
args.in_mult_KNet = 40
args.out_mult_KNet = 5

### training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 2000
args.n_batch = 100
args.lr = 1e-4
args.wd = 1e-4
args.CompositionLoss = True
args.alpha = 0.5

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0
chop = False
sequential_training = False
path_results = 'KNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data' + '/'
r2 = torch.tensor([1e-3]) # [10, 1, 0.1, 0.01, 1e-3]
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)

Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

traj_resultName = ['traj_lorDT_NLobs_rq3030_T20.pt']
dataFileName = ['data_lor_v0_rq3030_T20.pt']

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, h_nonlinear, R, args.T, args.T_test, m, n)# parameters for GT
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0
## Model with H=I          
sys_model_H = SystemModel(f, Q, h, R, args.T,args.T_test, m, n)
sys_model_H.InitSequence(m1x_0, m2x_0)

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input_long,train_target_long, cv_input, cv_target, test_input, test_target,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)   
if chop: 
   print("chop training data")    
   [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T)
   # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, T)
else:
   print("no chopping") 
   train_target = train_target_long[:,:,0:args.T]
   train_input = train_input_long[:,:,0:args.T] 
   # cv_target = cv_target[:,:,0:T]
   # cv_input = cv_input[:,:,0:T]  

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())

########################
### Evaluate Filters ###
########################
# ### Evaluate EKF full
print("Evaluate EKF full")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(args, sys_model, test_input, test_target)

# ### Save trajectories
# trajfolderName = 'Filters' + '/'
# DataResultName = traj_resultName[0]
# EKF_sample = torch.reshape(EKF_out[0],[1,m,args.T_test])
# target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
# input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
# torch.save({
#             'EKF': EKF_sample,
#             'ground_truth': target_sample,
#             'observation': input_sample,
#             }, trajfolderName+DataResultName)


##########################
### Evaluate KalmanNet ###
##########################
## Build Neural Network
print("KalmanNet start")
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model, args)
## Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
KalmanNet_Pipeline.setTrainingParams(args) 
if(chop):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,randomInit=True,train_init=train_init)
else:
   print("Composition Loss:",args.CompositionLoss)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)




