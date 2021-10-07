import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from KalmanNet_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_EKF import Pipeline_EKF

from KalmanNet_build import NNBuild
from KalmanNet_train import NNTrain
from KalmanNet_test import NNTest

from PF_test import PFTest
from KalmanNet_nn import KalmanNetNN

from datetime import datetime

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n,delta_t_gen,delta_t
from model import f, h, fInacc, hInacc, fRotate

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")


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

######################################
###  Compare EKF, RTS and RTSNet   ###
######################################
offset = 0
sequential_training = False
path_results = 'KNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/'
data_gen = 'data_gen.pt'
data_gen_file = torch.load(DatafolderName+data_gen, map_location=cuda0)
[true_sequence] = data_gen_file['All Data']

r = torch.tensor([1.])
lambda_q = torch.tensor([0.8])
traj_resultName = ['traj_lor_dec_PF_r0.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
# EKFResultName = 'EKF_obsmis_rq1030_T2000_NT100' 

for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("Search 1/q2 [dB]: ", 10 * torch.log10(1/lambda_q[rindex]**2))
   Q_mod = (lambda_q[rindex]**2) * torch.eye(m)
   R_mod = (r[rindex]**2) * torch.eye(n)
   # True Model
   sys_model_true = SystemModel(f, Q_mod, h, R_mod, T, T_test)
   sys_model_true.InitSequence(m1x_0, m2x_0)

   # Model with partial Info
   sys_model = SystemModel(fInacc, Q_mod, h, R_mod, T, T_test)
   sys_model.InitSequence(m1x_0, m2x_0)

   #Generate and load data Decimation case (chopped)
   print("Data Gen")
   [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r[rindex], offset)
   print("testset size:",test_target.size())
   [train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, r[rindex], offset)
   [cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_CV, h, r[rindex], offset)
    
   [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target_long.size())
   
   # Particle filter
   print("Start PF test")
   [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out, t_PF] = PFTest(sys_model_true, test_input, test_target, init_cond=None)
   print(f"MSE PF J=5: {MSE_PF_dB_avg} [dB] (T = {T_test})")
   [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial, t_PF] = PFTest(sys_model, test_input, test_target, init_cond=None)
   print(f"MSE PF J=2: {MSE_PF_dB_avg} [dB] (T = {T_test})")
   
   # KNet with model mismatch
   ## Build Neural Network
   Model = NNBuild(sys_model)
   ## Train Neural Network
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = NNTrain(sys_model, Model, cv_input_long, cv_target_long, train_input, train_target, path_results, sequential_training)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_KG_array, knet_out,RunTime] = NNTest(sys_model, test_input, test_target, path_results)
   # Print MSE Cross Validation
   print("MSE Test:", MSE_test_dB_avg, "[dB]")
   # Save trajectories
   trajfolderName = 'KNet' + '/'
   DataResultName = traj_resultName[rindex]
   torch.save({'PF J=5':PF_out,
               'PF J=2':PF_out_partial,
               # 'KNet': knet_out,
               }, trajfolderName+DataResultName)

   ## Save histogram
   # MSE_ResultName = 'Partial_MSE_KNet' 
   # torch.save(KNet_MSE_test_dB_avg,trajfolderName + MSE_ResultName)

   





