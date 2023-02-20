import torch
import torch.nn as nn
from datetime import datetime

import Filters.EKF_test as EKF_test

from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config
from Simulations.utils import Decimate_and_perturbate_Data,Short_Traj_Split
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,delta_t_gen,delta_t,\
f, h, h_nobatch, fInacc, Q_structure, R_structure

from Pipelines.Pipeline_EKF import Pipeline_EKF
from KNet.KalmanNet_nn import KalmanNetNN

from Plot import Plot_extended as Plot

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
args.N_CV = 10
args.N_T = 10
args.T = 3000
args.T_test = 3000
### training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 2000
args.n_batch = 8
args.lr = 1e-4
args.wd = 1e-4

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0 # offset for the data
chop = False # whether to chop the dataset sequences into smaller ones
path_results = 'KNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/'
DatafileName = 'decimated_r0_Ttest3000_NT10.pt'
data_gen = 'data_gen.pt'
data_gen_file = torch.load(DatafolderName+data_gen)
[true_sequence] = data_gen_file['All Data']

r = torch.tensor([1])
lambda_q = torch.tensor([0.3873])

print("1/r2 [dB]: ", 10 * torch.log10(1/r[0]**2))
print("Search 1/q2 [dB]: ", 10 * torch.log10(1/lambda_q[0]**2))
Q = (lambda_q[0]**2) * Q_structure
R = (r[0]**2) * R_structure 
# True Model
sys_model_true = SystemModel(f, Q, h, R, args.T, args.T_test,m,n)
sys_model_true.InitSequence(m1x_0, m2x_0)

# Model with partial Info
sys_model = SystemModel(fInacc, Q, h, R, args.T, args.T_test,m,n)
sys_model.InitSequence(m1x_0, m2x_0)

##############################################
### Generate and load data Decimation case ###
##############################################
########################
print("Data Gen")
########################
[test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, args.N_T, h_nobatch, r[0], offset) 
[train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, args.N_E, h_nobatch, r[0], offset)
[cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, args.N_CV, h_nobatch, r[0], offset)
if chop:
   print("chop training data")  
   [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T)
   args.N_E = train_target.size()[0]
else:
   print("no chopping") 
   train_target = train_target_long
   train_input = train_input_long
# Save dataset
if(chop):
   torch.save([train_input, train_target, train_init, cv_input_long, cv_target_long, test_input, test_target], DatafolderName+DatafileName)
else:
   torch.save([train_input, train_target, cv_input_long, cv_target_long, test_input, test_target], DatafolderName+DatafileName)

#########################
print("Data Load")
#########################
[train_input, train_target, cv_input_long, cv_target_long, test_input, test_target] = torch.load(DatafolderName+DatafileName, map_location=device)  
if(chop):
   print("chop training data")  
   [train_target, train_input, train_init] = Short_Traj_Split(train_target, train_input, args.T)
   args.N_E = train_target.size()[0]
print("load dataset to device:",train_input.device)
print("testset size:",test_target.size())
print("trainset size:",train_target.size())
print("cvset size:",cv_target_long.size())

########################################
### Evaluate Observation Noise Floor ###
########################################
args.N_T = len(test_input)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(args.N_T)# MSE [Linear]
for j in range(0, args.N_T):        
   MSE_obs_linear_arr[j] = loss_obs(test_input[j], test_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")
###################################################
args.N_E = len(train_input)
MSE_obs_linear_arr = torch.empty(args.N_E)# MSE [Linear]
for j in range(0, args.N_E):        
   MSE_obs_linear_arr[j] = loss_obs(train_input[j], train_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(train dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(train dataset) - STD:", obs_std_dB, "[dB]")

########################
### Evaluate Filters ###
########################
### EKF
print("Start EKF test J=5")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKF_test.EKFTest(args, sys_model_true, test_input, test_target)
print("Start EKF test J=2")
[MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKF_test.EKFTest(args, sys_model, test_input, test_target)

########################################
### KalmanNet with model mismatch ######
########################################
## Build Neural Network
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setssModel(sys_model)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
# Train Neural Network
KNet_Pipeline.setTrainingParams(args)
if(chop):
   KNet_Pipeline.NNTrain(sys_model,cv_input_long,cv_target_long,train_input,train_target,path_results,\
                         randomInit=True,train_init=train_init)
else:
   KNet_Pipeline.NNTrain(sys_model,cv_input_long,cv_target_long,train_input,train_target,path_results)
# Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, knet_out,t] = KNet_Pipeline.NNTest(sys_model,test_input,test_target,path_results)



# Save trajectories
trajfolderName = 'Simulations/Lorenz_Atractor' + '/'
DataResultName = 'traj_lor_dec.pt'
target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
torch.save({           
            'True':target_sample,
            'Observation':input_sample,
            'EKF J=5':EKF_out,
            'EKF J=2':EKF_out_partial,                       
            'KNet': knet_out,
            }, trajfolderName+DataResultName)

#############
### Plot  ###
#############
titles = ["True Trajectory","Observation","EKF","KNet"]
input = [target_sample,input_sample,EKF_out_partial, knet_out]
Net_Plot = Plot(trajfolderName,DataResultName)
Net_Plot.plotTrajectories(input,3, titles,trajfolderName+"lor_dec_trajs.png")





