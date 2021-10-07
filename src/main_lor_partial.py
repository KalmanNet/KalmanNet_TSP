import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from Extended_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_EKF import Pipeline_EKF

from Extended_KalmanNet_nn import KalmanNetNN

from datetime import datetime

from Plot import Plot_extended as Plot

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n,delta_t_gen,delta_t
from model import f, h, fInacc, hInacc, fRotate, h_nonlinear

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
DatafolderName = 'Simulations/Lorenz_Atractor/data/v0_smallT_NT1000' + '/'
data_gen = 'data_gen.pt'
# data_gen_file = torch.load(DatafolderName+data_gen, map_location=cuda0)
# [true_sequence] = data_gen_file['All Data']

r2 = torch.tensor([1])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)

q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

r2optdB = torch.tensor([-1])
ropt = torch.sqrt(10**(-r2optdB/10))
print("Searched optimal 1/r2 [dB]: ", 10 * torch.log10(1/ropt**2))

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
dataFileName = ['r0q0_T20.pt']#,'data_lor_v20_r1e-2_T100.pt','data_lor_v20_r1e-3_T100.pt','data_lor_v20_r1e-4_T100.pt']
EKFResultName = 'EKF_nonLinearh_rq00_T20' 
for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q[rindex]**2))
   #Model
   sys_model = SystemModel(f, q[rindex], h, r[rindex], T, T_test, m, n,"Lor")
   sys_model.InitSequence(m1x_0, m2x_0)

  #  sys_model_partialf = SystemModel(fInacc, q[rindex], h, r[rindex], T, T_test, m, n,"Lor")
  #  sys_model_partialf.InitSequence(m1x_0, m2x_0)

  #  sys_model_partialf_optq = SystemModel(fInacc, qopt, h, r[rindex], T, T_test, m, n,'lor')
  #  sys_model_partialf_optq.InitSequence(m1x_0, m2x_0)

   sys_model_partialh = SystemModel(f, q[rindex], h_nonlinear, r[rindex], T, T_test, m, n,"Lor")
   sys_model_partialh.InitSequence(m1x_0, m2x_0)

   sys_model_partialh_optr = SystemModel(f, q[rindex], h_nonlinear, ropt, T, T_test, m, n,'lor')
   sys_model_partialh_optr.InitSequence(m1x_0, m2x_0)
   
   #Generate and load data DT case
   print("Start Data Gen")
   # T = 2000
   DataGen(sys_model, DatafolderName + dataFileName[rindex], T, T_test)
   print("Data Load")
   print(dataFileName[rindex])
   [train_input_long, train_target_long, cv_input, cv_target, test_input, test_target] =  torch.load(DatafolderName + dataFileName[rindex],map_location=cuda0)  
   print("trainset long:",train_target_long.size())
   # T = 100
   # [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   # print("trainset chopped:",train_target.size())
   print("testset:",test_target.size())
   print("cvset:",cv_target.size())
   
   #Generate and load data Decimation case (chopped)
   # print("Data Gen")
   # [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r[rindex], offset)
   # print(test_target.size())
   # [train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, r[rindex], offset)
   # [cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_CV, h, r[rindex], offset)

   # [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   # [cv_target, cv_input] = Short_Traj_Split(cv_target_long, cv_input_long, T)

   #Evaluate EKF true
   # [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
   # #Evaluate EKF partial (h or r)
   # [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialh, test_input, test_target)
   #Evaluate EKF partial optq
  #  [MSE_EKF_linear_arr_partialoptq, MSE_EKF_linear_avg_partialoptq, MSE_EKF_dB_avg_partialoptq, EKF_KG_array_partialoptq, EKF_out_partialoptq] = EKFTest(sys_model_partialf_optq, test_input, test_target)
  #  #Evaluate EKF partial optr
   [MSE_EKF_linear_arr_partialoptr, MSE_EKF_linear_avg_partialoptr, MSE_EKF_dB_avg_partialoptr, EKF_KG_array_partialoptr, EKF_out_partialoptr] = EKFTest(sys_model_partialh_optr, test_input, test_target)
   

   
   # Save results

   # EKFfolderName = 'KNet' + '/'
   # torch.save({#'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
   # #             'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
   #             # 'MSE_EKF_linear_arr_partial': MSE_EKF_linear_arr_partial,
   #             # 'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partial,
   #             # 'MSE_EKF_linear_arr_partialoptr': MSE_EKF_linear_arr_partialoptr,
   #             # 'MSE_EKF_dB_avg_partialoptr': MSE_EKF_dB_avg_partialoptr,
   #             }, EKFfolderName+EKFResultName)

   # KNet without model mismatch
   # modelFolder = 'KNet' + '/'
   # KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
   # KNet_Pipeline.setssModel(sys_model)
   # KNet_model = KalmanNetNN()
   # KNet_model.Build(sys_model)
   # KNet_Pipeline.setModel(KNet_model)
   # KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=100, learningRate=5e-3, weightDecay=1e-4)

   # # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")

   # KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
   # [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
   # KNet_Pipeline.save()
   
   # KNet with model mismatch
   # modelFolder = 'KNet' + '/'
   # KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
   # KNet_Pipeline.setssModel(sys_model_partialh)
   # KNet_model = KalmanNetNN()
   # KNet_model.Build(sys_model_partialh)
   # KNet_Pipeline.setModel(KNet_model)
   # KNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=50, learningRate=1e-3, weightDecay=1e-9)

   # # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet_obsmis_rq1030_T2000.pt",map_location=cuda0)  

   # KNet_Pipeline.NNTrain(N_E, train_input_long, train_target_long, N_CV, cv_input, cv_target)
   # [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
   # KNet_Pipeline.save()

   # # Save trajectories
   # # trajfolderName = 'KNet' + '/'
   # # DataResultName = traj_resultName[rindex]
   # # # EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,T_test])
   # # # EKF_Partial_sample = torch.reshape(EKF_out_partial[0,:,:],[1,m,T_test])
   # # # target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
   # # # input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
   # # # KNet_sample = torch.reshape(KNet_test[0,:,:],[1,m,T_test])
   # # torch.save({
   # #             'KNet': KNet_test,
   # #             }, trajfolderName+DataResultName)

   # ## Save histogram
   # EKFfolderName = 'KNet' + '/'
   # torch.save({'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
   #             'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
   #             'MSE_EKF_linear_arr_partial': MSE_EKF_linear_arr_partial,
   #             'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partial,
   #             # 'MSE_EKF_linear_arr_partialoptr': MSE_EKF_linear_arr_partialoptr,
   #             # 'MSE_EKF_dB_avg_partialoptr': MSE_EKF_dB_avg_partialoptr,
   #             'KNet_MSE_test_linear_arr': KNet_MSE_test_linear_arr,
   #             'KNet_MSE_test_dB_avg': KNet_MSE_test_dB_avg,
   #             }, EKFfolderName+EKFResultName)

   





