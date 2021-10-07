from Simulations.Lorenz_Atractor.model import fRotate
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import random
import torch.nn as nn
from EKF_test import EKFTest
from Extended_sysmdl import SystemModel
from Extended_data import DataGen, DataLoader_GPU, DataGen_True,Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_EKF import Pipeline_EKF

from Extended_KalmanNet_nn import KalmanNetNN

from datetime import datetime

from Plot import Plot_extended as Plot

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, lambda_q_mod, lambda_r_mod, m, n,delta_t_gen,delta_t
from model import f, h, fInacc,hInacc,fRotate

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
   device = torch.device("cpu")
   print("Running on the CPU")


# print("Start Data Gen")
# offset = 0
# r2 = torch.tensor([10])
# r = torch.sqrt(r2)
# q_gen = 0
# DatafolderName = 'Simulations/Lorenz_Atractor/data' + '/'
# data_gen = 'data_gen.pt'
# data_gen_file = torch.load(DatafolderName+data_gen, map_location=device)
# [true_sequence] = data_gen_file['All Data']
# [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r, offset)

# vdB = -20 # ratio v=q2/r2
# v = 10**(vdB/10)

# q2_gen = torch.mul(v,r2)
# q_gen = torch.sqrt(q2_gen)
# print("data obs noise 1/r2 [dB]: ", 10 * torch.log10(1/r**2))
# print("data process noise 1/q2 [dB]: ", 10 * torch.log10(1/q_gen**2))
# #Model
# sys_model = SystemModel(f, q_gen, h, r, T, T_test, m, n,"Lor")
# sys_model.InitSequence(m1x_0, m2x_0)
# #Generate and load data
# DataGen(sys_model, DatafolderName + dataFileName, T, T_test)
# print("Data Load")
# [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(DatafolderName + dataFileName)  
# print(test_target.size())  

# dataFileName_long = 'data_pen_highresol_q1e-5_long.pt'
# true_sequence = torch.load(dataFolderName + dataFileName_long, map_location=device)
# [test_target_zeroinit, test_input_zeroinit] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, lambda_r_mod, offset=0)
# test_target = torch.empty(N_T,m,T_test)
# test_input = torch.empty(N_T,n,T_test)
### Random init
# print("random init testing data") 
# for test_i in range(N_T):
#    rand_seed = random.randint(0,10000-T_test-1)
#    test_target[test_i,:,:] = test_target_zeroinit[test_i,:,rand_seed:rand_seed+T_test]
#    test_input[test_i,:,:] = test_input_zeroinit[test_i,:,rand_seed:rand_seed+T_test]
# test_target = test_target_zeroinit[:,:,0:T_test]
# test_input = test_input_zeroinit[:,:,0:T_test]

r2_gen = torch.tensor([1])
r_gen = torch.sqrt(r2_gen)
rindex = 0
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2_gen = torch.mul(v,r2_gen)
q_gen = torch.sqrt(q2_gen)
DatafolderName = 'Simulations/Lorenz_Atractor/data' + '/'
dataFileName = ['data_lor_v20_rq020_T1000.pt']#,'data_lor_v20_r1e-1_T2000.pt','data_lor_v20_r1e-2_T2000.pt']

sys_model = SystemModel(f, q_gen, h, r_gen, T, T_test, m, n,"Lor")
sys_model.InitSequence(m1x_0, m2x_0)

# [train_input_long, train_target_long, cv_input_long, cv_target_long, test_input, test_target] = DataLoader_GPU(DatafolderName + dataFileName[rindex])  
# test_input = test_input[50:51,:,:]
# test_target = test_target[50:51,:,:]

print("Start Data Gen")
T = 1000
DataGen(sys_model, DatafolderName + dataFileName[rindex], T, T_test)
print("Data Load")
[train_input_long, train_target_long, cv_input_long, cv_target_long, test_input, test_target] =  torch.load(DatafolderName + dataFileName[rindex],map_location=device)  
print("trainset long:",train_target_long.size())

print("testset:",test_target.size())

r2 = torch.tensor([1e-2])
r = torch.sqrt(r2)
print("data obs noise 1/r2 [dB]: ", 10 * torch.log10(1/r_gen**2))
print("data process noise 1/q2 [dB]: ", 10 * torch.log10(1/q_gen**2))
# dataFileName = ['data_pen_r1_1.pt','data_pen_r1_2.pt','data_pen_r1_3.pt','data_pen_r1_4.pt','data_pen_r1_5.pt']
for index in range(0, len(r)):

   #Model

   # sys_model_partialf = SystemModel(fInacc, q_gen, h, r_gen, T, T_test, m, n,'lor')
   # sys_model_partialf.InitSequence(m1x_0, m2x_0)

   # sys_model_partialf_optq = SystemModel(fInacc, q[index], h, r_gen, T, T_test, m, n,'lor')
   # sys_model_partialf_optq.InitSequence(m1x_0, m2x_0)

   sys_model_partialh = SystemModel(f, q_gen, hInacc, r_gen, T, T_test, m, n,'lor')
   sys_model_partialh.InitSequence(m1x_0, m2x_0)

   sys_model_partialh_optr = SystemModel(f, q_gen, hInacc, r[index], T, T_test, m, n,'lor')
   sys_model_partialh_optr.InitSequence(m1x_0, m2x_0)

   #Evaluate EKF True
   [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
   
   ### Search EKF process model mismatch
   # print("search 1/q2 [dB]: ", 10 * torch.log10(1/q[index]**2))
   # [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialf, test_input, test_target)
   # [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialf_optq, test_input, test_target)
   ### Search EKF observation model mismatch
   print("search 1/r2 [dB]: ", 10 * torch.log10(1/r[index]**2))
   [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialh, test_input, test_target)
   [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialh_optr, test_input, test_target)
  
