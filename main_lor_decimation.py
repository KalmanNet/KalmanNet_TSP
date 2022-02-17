import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn

from KalmanNet_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_EKF import Pipeline_EKF

from EKF_test import EKFTest
from PF_test import PFTest
from UKF_test import UKFTest
from KalmanNet_nn import KalmanNetNN
from Vanilla_rnn import Vanilla_RNN

from datetime import datetime

from filing_paths import path_model
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

########################################
###  Compare EKF, UKF, PF and KNet   ###
########################################
offset = 0
split = True
path_results = 'KNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/'
data_gen = 'data_gen.pt'
data_gen_file = torch.load(DatafolderName+data_gen, map_location=cuda0)
[true_sequence] = data_gen_file['All Data']

r = torch.tensor([1.])
EKF_qoptdB = torch.tensor([8.2391])
EKF_q2 = 10**(-EKF_qoptdB/10)
UKF_q2 = 0.5**2
traj_resultName = ['traj_lor_dec_PF_r0.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
# EKFResultName = 'EKF_obsmis_rq1030_T2000_NT100' 

for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
  #  print("Search 1/q2 [dB]: ", EKF_qoptdB)
   Q_mod = (EKF_q2) * torch.eye(m)
   R_mod = (r[rindex]**2) * torch.eye(n)
   # True Model
   sys_model_true = SystemModel(f, Q_mod, h, R_mod, T, T_test)
   sys_model_true.InitSequence(m1x_0, m2x_0)

   # Model with partial Info
   sys_model = SystemModel(fInacc, Q_mod, h, R_mod, T, T_test)
   sys_model.InitSequence(m1x_0, m2x_0)

   Q_mod_UKF = (UKF_q2) * torch.eye(m)
   # True Model
   sys_model_true_UKF = SystemModel(f, Q_mod_UKF, h, R_mod, T, T_test)
   sys_model_true_UKF.InitSequence(m1x_0, m2x_0)

   # Model with partial Info
   sys_model_UKF = SystemModel(fInacc, Q_mod_UKF, h, R_mod, T, T_test)
   sys_model_UKF.InitSequence(m1x_0, m2x_0)

   #Generate and load data Decimation case (chopped)
   print("Data Gen")
   [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r[rindex], offset)
   print("testset size:",test_target.size())
   [train_target, train_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, r[rindex], offset)
   [cv_target, cv_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_CV, h, r[rindex], offset)
   if split: 
      [train_target, train_input] = Short_Traj_Split(train_target, train_input, T)
      [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, T)
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   
   # EKF
  #  print("Start EKF test")
  #  [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out] = EKFTest(sys_model_true, test_input, test_target)
  #  [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, KG_array_partial, EKF_out_partial] = EKFTest(sys_model, test_input, test_target)
   
   # Particle filter
  #  print("Start PF test")
  #  [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out] = PFTest(sys_model_true, test_input, test_target)
  #  [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial] = PFTest(sys_model, test_input, test_target)
   
   # UKF
  #  print("Start UKF test")
  #  [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out] = UKFTest(sys_model_true_UKF, test_input, test_target,delta_t)
  #  [MSE_UKF_linear_arr_partial, MSE_UKF_linear_avg_partial, MSE_UKF_dB_avg_partial, UKF_out_partial] = UKFTest(sys_model_UKF, test_input, test_target,delta_t)
   
   # KNet with model mismatch
   ## Build Neural Network
   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model)
   print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
   # ## Train Neural Network
   # KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
   # KNet_Pipeline.setssModel(sys_model)
   # KNet_Pipeline.setModel(KNet_model)
   # KNet_Pipeline.setTrainingParams(n_Epochs=100, n_Batch=10, learningRate=1e-3, weightDecay=1e-6)
   # KNet_Pipeline.NNTrain(train_input, train_target,cv_input, cv_target)
   # ## Test Neural Network
   # [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(test_input, test_target)
   # KNet_Pipeline.save()

   # Vanilla RNN with model mismatch
   ## Build RNN
   RNN_model = Vanilla_RNN()
   RNN_model.Build(sys_model, fully_agnostic = False)
   print("Number of trainable parameters for RNN:",sum(p.numel() for p in RNN_model.parameters() if p.requires_grad))
   ## Train Neural Network
   RNN_Pipeline = Pipeline_EKF(strTime, "KNet", "VanillaRNN")
   RNN_Pipeline.setssModel(sys_model)
   RNN_Pipeline.setModel(RNN_model)
   RNN_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=10, learningRate=1e-2, weightDecay=1e-6)
   RNN_Pipeline.NNTrain(train_input, train_target,cv_input, cv_target)
   ## Test Neural Network
   [RNN_MSE_test_linear_arr, RNN_MSE_test_linear_avg, RNN_MSE_test_dB_avg, RNN_test] = RNN_Pipeline.NNTest(test_input, test_target)
   RNN_Pipeline.save()
   
   # Save trajectories
  #  trajfolderName = 'KNet' + '/'
  #  DataResultName = traj_resultName[rindex]
  #  torch.save({'PF J=5':PF_out,
  #              'PF J=2':PF_out_partial,
  #              # 'KNet': knet_out,
  #              }, trajfolderName+DataResultName)

   ## Save histogram
   # MSE_ResultName = 'Partial_MSE_KNet' 
   # torch.save(KNet_MSE_test_dB_avg,trajfolderName + MSE_ResultName)

   





