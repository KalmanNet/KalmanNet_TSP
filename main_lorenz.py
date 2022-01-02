import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn

from EKF_test import EKFTest
from UKF_test import UKFTest
from PF_test import PFTest

from Extended_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_EKF import Pipeline_EKF

from Extended_KalmanNet_nn import KalmanNetNN

from datetime import datetime

from Plot import Plot_extended as Plot

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n,delta_t_gen,delta_t
from model import f, h, fInacc, hInacc, fRotate, h_nonlinear

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
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

#########################
###  Set parameters   ###
#########################
offset = 0
chop = False
DatafolderName = 'Simulations/Lorenz_Atractor/data/T2000_NT100' + '/'
# data_gen = 'data_gen.pt'
# data_gen_file = torch.load(DatafolderName+data_gen, map_location=dev)
# [true_sequence] = data_gen_file['All Data']

r2 = torch.tensor([1,0.1,0.01,1e-3,1e-4])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)

q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

### q and r searched for filters
r2searchdB = torch.tensor([-5,0,5])
rsearch = torch.sqrt(10**(-r2searchdB/10))
q2searchdB = torch.tensor([20,15,10])
qsearch = torch.sqrt(10**(-q2searchdB/10))

### q and r optimized for filters
r2optdB = torch.tensor([3.0103])
ropt = torch.sqrt(10**(-r2optdB/10))

r2optdB_partial = torch.tensor([3.0103])
ropt_partial = torch.sqrt(10**(-r2optdB_partial/10))

q2optdB = torch.tensor([18.2391,28.2391,38.2391,48,55])
qopt = torch.sqrt(10**(-q2optdB/10))

q2optdB_partial = torch.tensor([18.2391,28.2391,38.2391,48,55])
qopt_partial = torch.sqrt(10**(-q2optdB_partial/10))

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
dataFileName = ['data_lor_v20_rq020_T2000.pt','data_lor_v20_rq1030_T2000.pt','data_lor_v20_rq2040_T2000.pt','data_lor_v20_rq3050_T2000.pt','data_lor_v20_rq4060_T2000.pt']# for T=2000
# dataFileName = ['data_lor_v20_rq020_T1000_NT100.pt','data_lor_v20_rq1030_T1000_NT100.pt','data_lor_v20_rq2040_T1000_NT100.pt','data_lor_v20_rq3050_T1000_NT100.pt']# for T=1000
EKFResultName = ['EKF_rq020_T2000','EKF_rq1030_T2000','EKF_rq2040_T2000','EKF_rq3050_T2000','EKF_rq4060_T2000'] 
UKFResultName = ['UKF_rq020_T2000','UKF_rq1030_T2000','UKF_rq2040_T2000','UKF_rq3050_T2000','UKF_rq4060_T2000'] 
PFResultName = ['PF_rq020_T2000','PF_rq1030_T2000','PF_rq2040_T2000','PF_rq3050_T2000','PF_rq4060_T2000'] 

for index in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[index]**2))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q[index]**2))
   
   #############################
   ### Prepare System Models ###
   #############################
   sys_model = SystemModel(f, q[index], h, r[index], T, T_test, m, n,"Lor")
   sys_model.InitSequence(m1x_0, m2x_0)

   sys_model_optq = SystemModel(f, qopt[index], h, r[index], T, T_test, m, n,"Lor")
   sys_model_optq.InitSequence(m1x_0, m2x_0)

   sys_model_partialf_optq = SystemModel(fInacc, qopt_partial[index], h, r[index], T, T_test, m, n,"Lor")
   sys_model_partialf_optq.InitSequence(m1x_0, m2x_0)

   # sys_model_partialh = SystemModel(f, q[index], h_nonlinear, r[index], T, T_test, m, n,"Lor")
   # sys_model_partialh.InitSequence(m1x_0, m2x_0)

   # sys_model_partialh_optr = SystemModel(f, q[index], h_nonlinear, ropt, T, T_test, m, n,'lor')
   # sys_model_partialh_optr.InitSequence(m1x_0, m2x_0)
   
   #################################
   ### Generate and load DT data ###
   #################################
   # print("Start Data Gen")
   # DataGen(sys_model, DatafolderName + dataFileName[index], T, T_test,randomInit=False)
   print("Data Load")
   print(dataFileName[index])
   [train_input_long,train_target_long, cv_input, cv_target, test_input, test_target] =  torch.load(DatafolderName + dataFileName[index],map_location=dev)  
   if chop: 
      print("chop training data")    
      [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
      # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, T)
   else:
      print("no chopping") 
      train_target = train_target_long[:,:,0:T]
      train_input = train_input_long[:,:,0:T] 
      # cv_target = cv_target[:,:,0:T]
      # cv_input = cv_input[:,:,0:T]  

   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())
   
   """
   ############################################################
   ### Generate and load data for decimation case (chopped) ###
   ############################################################
   print("Data Gen")
   [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r[rindex], offset)
   print(test_target.size())
   [train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, r[rindex], offset)
   [cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_CV, h, r[rindex], offset)

   [train_target, train_input] = Short_Traj_Split(train_target_long, train_input_long, T)
   [cv_target, cv_input] = Short_Traj_Split(cv_target_long, cv_input_long, T)
   """
   ################################
   ### Evaluate EKF, UKF and PF ###
   ################################
   ### grid search of opt q for benchmarks
   # for searchindex in range(0, len(qsearch)):
   #    print("\n Searched optimal 1/q2 [dB]: ", 10 * torch.log10(1/qsearch[searchindex]**2))

   #    sys_model_searchq = SystemModel(f, qsearch[searchindex], h, r[index], T, T_test, m, n,"Lor")
   #    sys_model_searchq.InitSequence(m1x_0, m2x_0)
   #    print("Evaluate EKF true")
   #    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model_searchq, test_input, test_target)
      
   #    print("Evaluate UKF True")
   #    [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out] = UKFTest(sys_model_searchq, test_input, test_target)
      
   #    print("Evaluate PF True")
   #    [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out] = PFTest(sys_model_searchq, test_input, test_target)
   
   #    # Filters only have partial info of process model
   #    sys_model_partialf_searchq = SystemModel(fInacc, qsearch[searchindex], h, r[index], T, T_test, m, n,'lor')
   #    sys_model_partialf_searchq.InitSequence(m1x_0, m2x_0)
   #    print("Evaluate EKF Partial")
   #    [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partialf_searchq, test_input, test_target)
      
   #    print("Evaluate UKF Partial")
   #    [MSE_UKF_linear_arr_partial, MSE_UKF_linear_avg_partial, MSE_UKF_dB_avg_partial, UKF_out_partial] = UKFTest(sys_model_partialf_searchq, test_input, test_target)
   
   #    print("Evaluate PF Partial")
   #    [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial] = PFTest(sys_model_partialf_searchq, test_input, test_target)

   ### grid search of opt r for benchmarks
   # for searchindex in range(0, len(rsearch)):
   #    print("\n Searched optimal 1/r2 [dB]: ", 10 * torch.log10(1/rsearch[searchindex]**2))

   #    sys_model_searchr = SystemModel(f, q[index], h, rsearch[searchindex], T, T_test, m, n,"Lor")
   #    sys_model_searchr.InitSequence(m1x_0, m2x_0)
   #    print("Evaluate EKF true")
   #    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model_searchr, test_input, test_target)
      
   #    print("Evaluate UKF True")
   #    [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out] = UKFTest(sys_model_searchr, test_input, test_target)
      
   #    print("Evaluate PF True")
   #    [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out] = PFTest(sys_model_searchr, test_input, test_target)
   
   #    # Filters only have partial info of observation model
   #    sys_model_partialh_searchr = SystemModel(f, q[index], hInacc, rsearch[searchindex], T, T_test, m, n,"Lor")
   #    sys_model_partialh_searchr.InitSequence(m1x_0, m2x_0)
   #    print("Evaluate EKF Partial")
   #    [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest( sys_model_partialh_searchr, test_input, test_target)
      
   #    print("Evaluate UKF Partial")
   #    [MSE_UKF_linear_arr_partial, MSE_UKF_linear_avg_partial, MSE_UKF_dB_avg_partial, UKF_out_partial] = UKFTest(sys_model_partialh_searchr, test_input, test_target)
   
   #    print("Evaluate PF Partial")
   #    [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial] = PFTest( sys_model_partialh_searchr, test_input, test_target)

   
   print("Evaluate EKF true")
   [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model_optq, test_input, test_target)
   
   print("Evaluate UKF True")
   [MSE_UKF_linear_arr, MSE_UKF_linear_avg, MSE_UKF_dB_avg, UKF_out] = UKFTest(sys_model_optq, test_input, test_target)
   
   print("Evaluate PF True")
   [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out] = PFTest(sys_model_optq, test_input, test_target)
   
   #Evaluate partial_f
   print("Evaluate EKF Partial")
   [MSE_EKF_linear_arr_partialf, MSE_EKF_linear_avg_partialf, MSE_EKF_dB_avg_partialf, EKF_KG_array_partialf, EKF_out_partialf] = EKFTest(sys_model_partialf_optq, test_input, test_target)
   
   print("Evaluate UKF Partial")
   [MSE_UKF_linear_arr_partialf, MSE_UKF_linear_avg_partialf, MSE_UKF_dB_avg_partialf, UKF_out_partialf] = UKFTest(sys_model_partialf_optq, test_input, test_target)

   print("Evaluate PF Partial")
   [MSE_PF_linear_arr_partialf, MSE_PF_linear_avg_partialf, MSE_PF_dB_avg_partialf, PF_out_partialf] = PFTest(sys_model_partialf_optq, test_input, test_target)

      
   
   # Save results

   FilterfolderName = 'Filters/DT case/histogram/procmis/T2000' + '/'
   torch.save({'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
               'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
               'EKF_out':EKF_out,
               'MSE_EKF_linear_arr_partial': MSE_EKF_linear_arr_partialf,
               'MSE_EKF_dB_avg_partial': MSE_EKF_dB_avg_partialf,
               'EKF_out_partial': EKF_out_partialf,
               # 'MSE_EKF_linear_arr_partialh': MSE_EKF_linear_arr_partialh,
               # 'MSE_EKF_dB_avg_partialh': MSE_EKF_dB_avg_partialh,
               }, FilterfolderName+EKFResultName[index])

   torch.save({'MSE_UKF_linear_arr': MSE_UKF_linear_arr,
               'MSE_UKF_dB_avg': MSE_UKF_dB_avg,
               'UKF_out':UKF_out,
               'MSE_UKF_linear_arr_partialf': MSE_UKF_linear_arr_partialf,
               'MSE_UKF_dB_avg_partialf': MSE_UKF_dB_avg_partialf,
               'UKF_out_partialf': UKF_out_partialf,
               # 'MSE_UKF_linear_arr_partialh': MSE_UKF_linear_arr_partialh,
               # 'MSE_UKF_dB_avg_partialh': MSE_UKF_dB_avg_partialh,
               }, FilterfolderName+UKFResultName[index])

   torch.save({'MSE_PF_linear_arr': MSE_PF_linear_arr,
               'MSE_PF_dB_avg': MSE_PF_dB_avg,
               'PF_out':PF_out,
               'MSE_PF_linear_arr_partialf': MSE_PF_linear_arr_partialf,
               'MSE_PF_dB_avg_partialf': MSE_PF_dB_avg_partialf,
               'PF_out_partialf': PF_out_partialf,
               # 'MSE_EKF_linear_arr_partialoptr': MSE_EKF_linear_arr_partialoptr,
               # 'MSE_EKF_dB_avg_partialoptr': MSE_EKF_dB_avg_partialoptr,
               }, FilterfolderName+PFResultName[index])
   
#####################
### Evaluate KNet ###
#####################
### KNet without model mismatch
# sys_model = SystemModel(f, q[0], h, r[0], T, T_test, m, n,"Lor")# arbitary q and r
# sys_model.InitSequence(m1x_0, m2x_0)
# print("KNet with full model info")
# modelFolder = 'KNet' + '/'
# KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
# KNet_Pipeline.setssModel(sys_model)
# KNet_model = KalmanNetNN()
# KNet_model.Build(sys_model)
# KNet_Pipeline.setModel(KNet_model)
# KNet_Pipeline.setTrainingParams(n_Epochs=200, n_Batch=10, learningRate=1e-3, weightDecay=1e-4)

# KNet_Pipeline.model = torch.load(modelFolder+"model_KNet.pt")

# KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
# [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
# KNet_Pipeline.save()

### KNet with model mismatch
# print("KNet with model mismatch")
# sys_model_partialh = SystemModel(f, q[0], hInacc, r[0], T, T_test, m, n,"Lor")# arbitary q and r
# sys_model_partialh.InitSequence(m1x_0, m2x_0)
# modelFolder = 'KNet' + '/'
# KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KNet")
# KNet_Pipeline.setssModel(sys_model_partialh)
# KNet_model = KalmanNetNN()
# KNet_model.Build(sys_model_partialh)
# KNet_Pipeline.setModel(KNet_model)
# KNet_Pipeline.setTrainingParams(n_Epochs=200, n_Batch=10, learningRate=1e-3, weightDecay=1e-4)

# # KNet_Pipeline.model = torch.load(modelFolder+"model_KNet_obsmis_rq1030_T2000.pt",map_location=dev)  

# KNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
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

   





