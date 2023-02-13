import torch
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1E4
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.signal import find_peaks

# Legend
Klegend = ["Unsupervised KalmanNet - Train", "Unsupervised KalmanNet - Validation", "Unsupervised KalmanNet - Test", "Kalman Filter"]
RTSlegend = ["RTSNet - Train", "RTSNet - Validation", "RTSNet - Test", "RTS Smoother","Kalman Filter"]
ERTSlegend = ["RTSNet - Train","RTSNet - Validation", "RTSNet - Test", "RTS","EKF"]
error_evol = ["KNet Empirical Error","KNet Covariance Trace","KF Empirical Error","KF Covariance Trace","KNet Error Deviation","EKF Error Deviation"]
# Color
KColor = ['-ro','darkorange','k-', 'b-','g-']
RTSColor = ['red','darkorange','g-', 'b-']

class Plot_KF:
    
    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName

    def NNPlot_epochs(self, N_Epochs_plt, MSE_KF_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):

        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=Klegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])
        
        plt.xticks(fontsize= fontSize)
        plt.yticks(fontsize= fontSize)
        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Iterations', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.grid(True)
        # plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)


    def KFPlot(res_grid):

        plt.figure(figsize = (50, 20))
        x_plt = [-6, 0, 6]

        plt.plot(x_plt, res_grid[0][:], 'xg', label='minus')
        plt.plot(x_plt, res_grid[1][:], 'ob', label='base')
        plt.plot(x_plt, res_grid[2][:], '+r', label='plus')
        plt.plot(x_plt, res_grid[3][:], 'oy', label='base NN')

        plt.legend()
        plt.xlabel('Noise', fontsize=16)
        plt.ylabel('MSE Loss Value [dB]', fontsize=16)
        plt.title('Change', fontsize=16)
        plt.savefig('plt_grid_dB')

        print("\ndistribution 1")
        print("Kalman Filter")
        print(res_grid[0][0], "[dB]", res_grid[1][0], "[dB]", res_grid[2][0], "[dB]")
        print(res_grid[1][0] - res_grid[0][0], "[dB]", res_grid[2][0] - res_grid[1][0], "[dB]")
        print("KalmanNet", res_grid[3][0], "[dB]", "KalmanNet Diff", res_grid[3][0] - res_grid[1][0], "[dB]")

        print("\ndistribution 2")
        print("Kalman Filter")
        print(res_grid[0][1], "[dB]", res_grid[1][1], "[dB]", res_grid[2][1], "[dB]")
        print(res_grid[1][1] - res_grid[0][1], "[dB]", res_grid[2][1] - res_grid[1][1], "[dB]")
        print("KalmanNet", res_grid[3][1], "[dB]", "KalmanNet Diff", res_grid[3][1] - res_grid[1][1], "[dB]")

        print("\ndistribution 3")
        print("Kalman Filter")
        print(res_grid[0][2], "[dB]", res_grid[1][2], "[dB]", res_grid[2][2], "[dB]")
        print(res_grid[1][2] - res_grid[0][2], "[dB]", res_grid[2][2] - res_grid[1][2], "[dB]")
        print("KalmanNet", res_grid[3][2], "[dB]", "KalmanNet Diff", res_grid[3][2] - res_grid[1][2], "[dB]")

    def NNPlot_test(MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg,
               MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg):


        N_Epochs_plt = 100

        ###############################
        ### Plot per epoch [linear] ###
        ###############################
        plt.figure(figsize = (50, 20))

        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        y_plt3 = MSE_test_linear_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_linear_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend()
        plt.xlabel('Number of Training Epochs', fontsize=16)
        plt.ylabel('MSE Loss Value [linear]', fontsize=16)
        plt.title('MSE Loss [linear] - per Epoch', fontsize=16)
        plt.savefig('plt_model_test_linear')

        ###########################
        ### Plot per epoch [dB] ###
        ###########################
        plt.figure(figsize = (50, 20))

        x_plt = range(0, N_Epochs_plt)

        # KNet - Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=Klegend[2])

        # KF
        y_plt4 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=Klegend[3])

        plt.legend()
        plt.xlabel('Number of Training Epochs', fontsize=16)
        plt.ylabel('MSE Loss Value [dB]', fontsize=16)
        plt.title('MSE Loss [dB] - per Epoch', fontsize=16)
        plt.savefig('plt_model_test_dB')

        ########################
        ### Linear Histogram ###
        ########################
        plt.figure(figsize=(50, 20))
        sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
        sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        plt.title("Histogram [Linear]")
        plt.savefig('plt_hist_linear')

        fig, axes = plt.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
        sns.distplot(MSE_test_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label='KalmanNet', ax=axes[0])
        sns.distplot(MSE_KF_linear_arr, hist=False, kde=True, kde_kws={'linewidth': 3}, color='b', label='Kalman Filter', ax=axes[1])
        plt.title("Histogram [Linear]")
        plt.savefig('plt_hist_linear_1')

        ####################
        ### dB Histogram ###
        ####################

        plt.figure(figsize=(50, 20))
        sns.distplot(10 * torch.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet')
        sns.distplot(10 * torch.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter')
        plt.title("Histogram [dB]")
        plt.savefig('plt_hist_dB')


        fig, axes = plt.subplots(2, 1, figsize=(50, 20), sharey=True, dpi=100)
        sns.distplot(10 * torch.log10(MSE_test_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color='g', label = 'KalmanNet', ax=axes[0])
        sns.distplot(10 * torch.log10(MSE_KF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Kalman Filter', ax=axes[1])
        plt.title("Histogram [dB]")
        plt.savefig('plt_hist_dB_1')

        print('End')

class Plot_RTS(Plot_KF):

    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName

    def NNPlot_epochs(self, N_MiniBatchTrain_plt, BatchSize, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):
        N_Epochs_plt = np.floor(N_MiniBatchTrain_plt/BatchSize).astype(int) # number of epochs
        
        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=RTSlegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=RTSlegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=RTSlegend[2])

        # RTS
        y_plt4 = MSE_RTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, "g", label=RTSlegend[3])

        # KF
        y_plt5 = MSE_KF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, "orange", label=RTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)


    def NNPlot_Hist(self, MSE_KF_linear_arr, MSE_RTS_data_linear_arr, MSE_RTSNet_linear_arr):

        fileName = self.folderName + 'plt_hist_dB'
        fontSize = 32
        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(10, 25))
        ax = sns.displot(
            {self.modelName: 10 * torch.log10(MSE_RTSNet_linear_arr), 
            'Kalman Filter': 10 * torch.log10(MSE_KF_linear_arr),
            'RTS Smoother': 10 * torch.log10(MSE_RTS_data_linear_arr)},  # Use a dict to assign labels to each curve
            kind="kde",
            common_norm=False,  # Normalize each distribution independently: the area under each curve equals 1.
            palette=["blue", "orange", "g"],  # Use palette for multiple colors
            linewidth= 1,
        )
        plt.title(self.modelName + ":" +"Histogram [dB]")
        plt.xlabel('MSE Loss Value [dB]')
        plt.ylabel('Percentage')
        sns.move_legend(ax, "upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fileName)

    def KF_RTS_Plot_Linear(self, r, MSE_KF_RTS_dB,PlotResultName):
        fileName = self.folderName + PlotResultName
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_KF_RTS_dB[0,:], '-^',color='orange',linewidth=1, markersize=12, label=r'2x2, KF')
        plt.plot(x_plt, MSE_KF_RTS_dB[1,:], '--go',markerfacecolor='none',linewidth=3, markersize=12, label=r'2x2, RTS')
        plt.plot(x_plt, MSE_KF_RTS_dB[2,:], '-bo',linewidth=1, markersize=12, label=r'2x2, RTSNet')

        plt.legend(fontsize=32)
        plt.xlabel(r'Noise $\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        # plt.title('Comparing Kalman Filter and RTS Smoother', fontsize=32)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.savefig(fileName)

    def rotate_RTS_Plot_F(self, r, MSE_RTS_dB,rotateName):
        fileName = self.folderName + rotateName
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_RTS_dB[0,:], '-r^', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB[1,:], '-gx', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB[2,:], '-bo', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{F}_{\alpha=10^\circ}$)')

        plt.legend(fontsize=16)
        plt.xlabel(r'Noise $\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.savefig(fileName)  

    def rotate_RTS_Plot_H(self, r, MSE_RTS_dB,rotateName):
        fileName = self.folderName + rotateName
        magnifying_glass, main_H = plt.subplots(figsize = [25, 10])
        # main_H = plt.figure(figsize = [25, 10])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_H.plot(x_plt, NoiseFloor, '--r', linewidth=2, markersize=12, label=r'Noise Floor')
        main_H.plot(x_plt, MSE_RTS_dB[0,:], '-g^', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB] , 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        main_H.plot(x_plt, MSE_RTS_dB[1,:], '-yx', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        main_H.plot(x_plt, MSE_RTS_dB[2,:], '-bo', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')

        main_H.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-20, 15))
        main_H.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)

        ax2 = plt.axes([.15, .15, .27, .27]) 
        x1, x2, y1, y2 =  -0.2, 0.2, -5, 8
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=2, markersize=12, label=r'Noise Floor')
        ax2.plot(x_plt, MSE_RTS_dB[0,:], '-g^', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB] , 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        ax2.plot(x_plt, MSE_RTS_dB[1,:], '-yx', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        ax2.plot(x_plt, MSE_RTS_dB[2,:], '-bo', linewidth=2, markersize=12, label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')
        ax2.grid(True)
        plt.savefig(fileName)    

    def rotate_RTS_Plot_FHCompare(self, r, MSE_RTS_dB_F,MSE_RTS_dB_H,rotateName):
        fileName = self.folderName + rotateName
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r)

        plt.plot(x_plt, MSE_RTS_dB_F[0,:], '-r^', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_F[1,:], '-gx', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_F[2,:], '-bo', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{F}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[0,:], '--r^', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=0^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[1,:], '--gx', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTS Smoother ($\mathbf{H}_{\alpha=10^\circ}$)')
        plt.plot(x_plt, MSE_RTS_dB_H[2,:], '--bo', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], 2x2, RTSNet ($\mathbf{H}_{\alpha=10^\circ}$)')

        plt.legend(fontsize=16)
        plt.xlabel(r'Noise $\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)
        plt.savefig(fileName)  

    def plotTraj_CA(self,test_target, RTS_out, rtsnet_out, dim, file_name):
        legend = ["RTSNet", "Ground Truth", "MB RTS"]
        font_size = 14
        T_test = rtsnet_out[0].size()[1]
        x_plt = range(0, T_test)
        if dim==0:#position
            plt.plot(x_plt, rtsnet_out[0][0,:].detach().numpy(), label=legend[0])
            plt.plot(x_plt, test_target[0][0,:].detach().numpy(), label=legend[1])
            plt.plot(x_plt, RTS_out[0][0,:], label=legend[2])
            plt.legend(fontsize=font_size)
            plt.xlabel('t', fontsize=font_size)
            plt.ylabel('position', fontsize=font_size)
            plt.savefig(file_name) 
            plt.clf()
        elif dim==1:#velocity
            plt.plot(x_plt, rtsnet_out[0][1,:].detach().numpy(), label=legend[0])
            plt.plot(x_plt, test_target[0][1,:].detach().numpy(), label=legend[1])
            plt.plot(x_plt, RTS_out[0][1,:], label=legend[2])
            plt.legend(fontsize=font_size)
            plt.xlabel('t', fontsize=font_size)
            plt.ylabel('velocity', fontsize=font_size)
            plt.savefig(file_name)
            plt.clf()
        elif dim==2:#acceleration
            plt.plot(x_plt, rtsnet_out[0][2,:].detach().numpy(), label=legend[0])
            plt.plot(x_plt, test_target[0][2,:].detach().numpy(), label=legend[1])
            plt.plot(x_plt, RTS_out[0][2,:], label=legend[2])
            plt.legend(fontsize=font_size)
            plt.xlabel('t', fontsize=font_size)
            plt.ylabel('acceleration', fontsize=font_size)
            plt.savefig(file_name)
            plt.clf()
        else:
            print("invalid dimension")

class Plot_extended(Plot_RTS):
    def EKFPlot_Hist(self, MSE_EKF_linear_arr):   
        fileName = self.folderName + 'plt_hist_dB'
        fontSize = 32
        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))       
        sns.distplot(10 * np.log10(MSE_EKF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'b', label = 'Extended Kalman Filter')
        plt.title(self.modelName + ":" +"Histogram [dB]",fontsize=fontSize)
        plt.legend(fontsize=fontSize)
        plt.savefig(fileName)

    def KF_RTS_Plot(self, r, MSE_KF_RTS_dB):
        fileName = self.folderName + 'Nonlinear_KF_RTS_Compare_dB'
        plt.figure(figsize = (25, 10))
        x_plt = 10 * torch.log10(1/r**2)

        plt.plot(x_plt, MSE_KF_RTS_dB[0,:], '-gx', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], Toy Model, EKF')
        plt.plot(x_plt, MSE_KF_RTS_dB[1,:], '--bo', label=r'$\mathrm{\frac{q^2}{r^2}}=0$ [dB], Toy Model, Extended RTS')

        plt.legend(fontsize=32)
        plt.xlabel(r'Noise $\mathrm{\frac{1}{q^2}}$ [dB]', fontsize=32)
        plt.ylabel('MSE [dB]', fontsize=32)
        plt.title('Comparing Extended Kalman Filter and Extended RTS Smoother', fontsize=32)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.savefig(fileName)

    def NNPlot_trainsteps(self, N_MiniBatchTrain_plt, MSE_EKF_dB_avg, MSE_ERTS_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):
        N_Epochs_plt = N_MiniBatchTrain_plt
        
        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=ERTSlegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[range(0, N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=ERTSlegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=ERTSlegend[2])

        # RTS
        y_plt4 = MSE_ERTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=ERTSlegend[3])

        # EKF
        y_plt5 = MSE_EKF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, KColor[4], label=ERTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Steps', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.grid(True)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Step", fontsize=fontSize)
        plt.savefig(fileName)


    
    def NNPlot_epochs(self, N_E,N_MiniBatchTrain_plt, BatchSize, MSE_EKF_dB_avg, MSE_ERTS_dB_avg,
                      MSE_test_dB_avg, MSE_cv_dB_epoch, MSE_train_dB_epoch):
        N_Epochs_plt = np.floor(N_MiniBatchTrain_plt*BatchSize/N_E).astype(int) # number of epochs
        print(N_Epochs_plt)
        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train
        y_plt1 = MSE_train_dB_epoch[np.linspace(0,N_MiniBatchTrain_plt-1,N_Epochs_plt)]
        plt.plot(x_plt, y_plt1, KColor[0], label=ERTSlegend[0])

        # CV
        y_plt2 = MSE_cv_dB_epoch[np.linspace(0,N_MiniBatchTrain_plt-1,N_Epochs_plt)]
        plt.plot(x_plt, y_plt2, KColor[1], label=ERTSlegend[1])

        # Test
        y_plt3 = MSE_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt3, KColor[2], label=ERTSlegend[2])

        # RTS
        y_plt4 = MSE_ERTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt4, KColor[3], label=ERTSlegend[3])

        # EKF
        y_plt5 = MSE_EKF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, KColor[4], label=ERTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.grid(True)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.savefig(fileName)

    def NNPlot_Hist(self, MSE_EKF_linear_arr, MSE_ERTS_data_linear_arr, MSE_RTSNet_linear_arr):
    
        fileName = self.folderName + 'plt_hist_dB'
        fontSize = 32
        ####################
        ### dB Histogram ###
        ####################
        plt.figure(figsize=(25, 10))
        # sns.distplot(10 * torch.log10(MSE_RTSNet_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 5}, color='b', label = self.modelName)
        # sns.distplot(10 * torch.log10(MSE_EKF_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3}, color= 'orange', label = 'EKF')
        # sns.distplot(10 * torch.log10(MSE_ERTS_data_linear_arr), hist=False, kde=True, kde_kws={'linewidth': 3.2,"linestyle":'--'},color= 'g', label = 'RTS')
       
        # plt.title(self.modelName + ":" +"Histogram [dB]",fontsize=fontSize)
        # plt.legend(fontsize=fontSize)
        # plt.xlabel('MSE Loss Value [dB]', fontsize=fontSize)
        # plt.ylabel('Percentage', fontsize=fontSize)
        # plt.tick_params(labelsize=fontSize)
        # plt.grid(True)
        # plt.savefig(fileName)
        ax = sns.displot(
            {self.modelName: 10 * torch.log10(MSE_RTSNet_linear_arr), 
            'Kalman Filter': 10 * torch.log10(MSE_EKF_linear_arr),
            'RTS Smoother': 10 * torch.log10(MSE_ERTS_data_linear_arr)},  # Use a dict to assign labels to each curve
            kind="kde",
            common_norm=False,  # Normalize each distribution independently: the area under each curve equals 1.
            palette=["blue", "orange", "g"],  # Use palette for multiple colors
            linewidth= 1,
        )
        plt.title(self.modelName + ":" +"Histogram [dB]")
        plt.xlabel('MSE Loss Value [dB]')
        plt.ylabel('Percentage')
        sns.move_legend(ax, "upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fileName)

    def NNPlot_epochs_KF_RTS(self, N_MiniBatchTrain_plt, BatchSize, MSE_EKF_dB_avg, MSE_ERTS_dB_avg,
                      MSE_KNet_test_dB_avg, MSE_KNet_cv_dB_epoch, MSE_KNet_train_dB_epoch,
                      MSE_RTSNet_test_dB_avg, MSE_RTSNet_cv_dB_epoch, MSE_RTSNet_train_dB_epoch):
        N_Epochs_plt = np.floor(N_MiniBatchTrain_plt/BatchSize).astype(int) # number of epochs

        # File Name
        fileName = self.folderName + 'plt_epochs_dB'

        fontSize = 32

        # Figure
        plt.figure(figsize = (25, 10))

        # x_axis
        x_plt = range(0, N_Epochs_plt)

        # Train KNet and RTSNet
        # y_plt1 = MSE_KNet_train_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        # plt.plot(x_plt, y_plt1, KColor[0], label=Klegend[0])
        # y_plt2 = MSE_RTSNet_train_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        # plt.plot(x_plt, y_plt2, color=RTSColor[0],linestyle='-', marker='o', label=ERTSlegend[0])

        # CV KNet and RTSNet
        y_plt3 = MSE_KNet_cv_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt3, color=RTSColor[0],linestyle='-', marker='o', label=Klegend[1])
        y_plt4 = MSE_RTSNet_cv_dB_epoch[np.linspace(0,BatchSize*(N_Epochs_plt-1) ,N_Epochs_plt)]
        plt.plot(x_plt, y_plt4, color=RTSColor[1],linestyle='-', marker='o', label=ERTSlegend[1])

        # Test KNet and RTSNet
        y_plt5 = MSE_KNet_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt5, color=RTSColor[0],linestyle='--', label=Klegend[2])
        y_plt6 = MSE_RTSNet_test_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt6,color=RTSColor[1],linestyle='--', label=ERTSlegend[2])

        # RTS
        y_plt7 = MSE_ERTS_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt7, RTSColor[2], label=ERTSlegend[3])

        # EKF
        y_plt8 = MSE_EKF_dB_avg * torch.ones(N_Epochs_plt)
        plt.plot(x_plt, y_plt8, RTSColor[3], label=ERTSlegend[4])

        plt.legend(fontsize=fontSize)
        plt.xlabel('Number of Training Epochs', fontsize=fontSize)
        plt.ylabel('MSE Loss Value [dB]', fontsize=fontSize)
        plt.title(self.modelName + ":" + "MSE Loss [dB] - per Epoch", fontsize=fontSize)
        plt.grid(True)
        plt.savefig(fileName)

    def plotTrajectories(self,inputs, dim, titles, file_name):
    
        fig = plt.figure(figsize=(15, 10))
        plt.Axes (fig, [0,0,1,1])
        # plt.subplots_adjust(wspace=-0.2, hspace=-0.2)
        matrix_size = int(np.ceil(np.sqrt(len(inputs))))
        #gs1 = gridspec.GridSpec(matrix_size,matrix_size)
        gs1 = gridspec.GridSpec(3,2)
        gs1.update(wspace=0, hspace=0)
        gs2 = gridspec.GridSpec(5,1)
        gs2.update(wspace=0, hspace=1)
        plt.rcParams["figure.frameon"] = False
        plt.rcParams["figure.constrained_layout.use"]= True
        i=0
        for title in titles:
            inputs_numpy = inputs[i][0].detach().numpy()
            # gs1.update(wspace=-0.3,hspace=-0.3)
            if(dim==3):
                plt.rcParams["figure.frameon"] = False
                ax = fig.add_subplot(gs1[i],projection='3d')
                # if(i<3):
                #     ax = fig.add_subplot(gs1[i],projection='3d')
                # else:
                #     ax = fig.add_subplot(gs1[i:i+2],projection='3d')

                y_al = 0.73
                if(title == "True Trajectory"):
                    c = 'k'
                elif(title == "Observation"):
                    c = 'r'
                elif(title == "Extended RTS"):
                    c = 'b'
                    y_al = 0.68
                elif(title == "RTSNet"):
                    c = 'g'
                elif(title == "Particle Smoother"):
                    c = 'c'
                elif(title == "Vanilla RNN"):
                    c = 'm'
                elif(title == "KNet"):
                    c = 'y'               
                else:
                    c = 'purple'
                    y_al = 0.68

                ax.set_axis_off()
                ax.set_title(title, y=y_al, fontdict={'fontsize': 15,'fontweight' : 20,'verticalalignment': 'baseline'})
                ax.plot(inputs_numpy[0,:], inputs_numpy[1,:], inputs_numpy[2,:], c, linewidth=0.5)

                ## Plot display 
                #ax.set_yticklabels([])
                #ax.set_xticklabels([])
                #ax.set_zticklabels([])
                #ax.set_xlabel('x')
                #ax.set_ylabel('y')
                #ax.set_zlabel('z')

            if(dim==2):
                ax = fig.add_subplot(matrix_size, matrix_size,i+1)
                ax.plot(inputs_numpy[0,:],inputs_numpy[1,:], 'b', linewidth=0.75)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_title(title, pad=10, fontdict={'fontsize': 20,'fontweight' : 20,'verticalalignment': 'baseline'})

            if(dim==4):
                if(title == "True Trajectory"):
                    target_theta_sample = inputs_numpy[0,0,:]
        
                # ax = fig.add_subplot(matrix_size, matrix_size,i+1)
                ax = fig.add_subplot(gs2[i,:])
                # print(inputs_numpy[0,0,:])
                ax.plot(np.arange(np.size(inputs_numpy[0,:],axis=1)), inputs_numpy[0,0,:], 'b', linewidth=0.75)
                if(title != "True Trajectory"):
                    diff = target_theta_sample - inputs_numpy[0,0,:]
                    peaks, _ = find_peaks(diff, prominence=0.31)
                    troughs, _ = find_peaks(-diff, prominence=0.31)
                    for peak, trough in zip(peaks, troughs):
                        plt.axvspan(peak, trough, color='red', alpha=.2)
                # zoomed in
                # ax.plot(np.arange(20), inputs_numpy[0,0,0:20], 'b', linewidth=0.75)inputs_numpy[0,0,:]
                ax.set_xlabel('time [s]')
                ax.set_ylabel('theta [rad]')
                ax.set_title(title, pad=10, fontdict={'fontsize': 20,'fontweight' : 20,'verticalalignment': 'baseline'})

            i +=1
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=1000)

    def Partial_Plot_Lor(self, r, MSE_Partial_dB):
        fileName = self.folderName + 'Nonlinear_Lor_Partial_J=2'
        magnifying_glass, main_partial = plt.subplots(figsize = [20, 15])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_partial.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12, label=r'Noise Floor')
        main_partial.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12, label=r'EKF:  $\rm J_{mdl}=5$')
        main_partial.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12, label=r'EKF:  $\rm J_{mdl}=2$')
        main_partial.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12, label=r'RTS:  $\rm J_{mdl}=5$')
        main_partial.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12, label=r'RTS:  $ \rm J_{mdl}=2$')
        main_partial.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12, label=r'RTSNet: $ \rm J_{mdl}=2$')

        main_partial.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-60, 10))
        main_partial.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)
        
        ax2 = plt.axes([.15, .15, .25, .25]) 
        x1, x2, y1, y2 =  19.5, 20.5, -35, -10
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12)
        ax2.grid(True)          
        plt.savefig(fileName)

    
        fileName = self.folderName + 'Nonlinear_Pen_PartialF'
        magnifying_glass, main_partial = plt.subplots(figsize = [20, 15])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_partial.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12, label=r'Noise Floor')
        main_partial.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=4, markersize=12, label=r'EKF:  $\rm L=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=4, markersize=12, label=r'EKF:  $\rm L=1.1$')
        main_partial.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=2, markersize=12, label=r'RTS:  $\rm L=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12, label=r'RTS:  $ \rm L=1.1$')
        main_partial.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=24, label=r'RTSNet: $ \rm L=1.1$')

        main_partial.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-75, 5))
        main_partial.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)
        
        ax2 = plt.axes([.15, .15, .25, .25]) 
        x1, x2, y1, y2 =  19.5, 20.5, -55, -15
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12)
        ax2.grid(True)          
        plt.savefig(fileName)

    def Partial_Plot_H1(self, r, MSE_Partial_dB):
        fileName = self.folderName + 'Nonlinear_Lor_Partial_Hrot1'
        magnifying_glass, main_partial = plt.subplots(figsize = [20, 15])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_partial.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12, label=r'Noise Floor')
        main_partial.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12, label=r'EKF:  $\Delta{\theta}=0$')
        main_partial.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12, label=r'EKF:  $\Delta{\theta}=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12, label=r'RTS:  $\Delta{\theta}=0$')
        main_partial.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12, label=r'RTS:  $\Delta{\theta}=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12, label=r'RTSNet: $\Delta{\theta}=1$')

        main_partial.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-60, 10))
        main_partial.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)
        
        ax2 = plt.axes([.15, .15, .25, .25]) 
        x1, x2, y1, y2 =  19.5, 20.5, -35, -10
        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)
        ax2.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[0,:], '-yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[1,:], '--yx', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[2,:], '-bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[3,:], '--bo', linewidth=3, markersize=12)
        ax2.plot(x_plt, MSE_Partial_dB[4,:], '--g^', linewidth=3, markersize=12)
        ax2.grid(True)          
        plt.savefig(fileName)

    def Partial_Plot_KNetRTSNet_Compare(self, r, MSE_Partial_dB):
        fileName = self.folderName + 'Nonlinear_Lor_Partial_Hrot1_Compare'
        magnifying_glass, main_partial = plt.subplots(figsize = [20, 15])
        x_plt = 10 * torch.log10(1/r**2)
        NoiseFloor = -x_plt
        main_partial.plot(x_plt, NoiseFloor, '--r', linewidth=3, markersize=12, label=r'Noise Floor')
        main_partial.plot(x_plt, MSE_Partial_dB[0,:], '--bo', linewidth=3, markersize=12, label=r'KNet: $\Delta{\theta}=1$')
        main_partial.plot(x_plt, MSE_Partial_dB[1,:], '--g^', linewidth=3, markersize=12, label=r'RTSNet: $\Delta{\theta}=1$')

        main_partial.set(xlim=(x_plt[0], x_plt[len(x_plt)-1]), ylim=(-60, 10))
        main_partial.legend(fontsize=20)
        plt.xlabel(r'$\mathrm{\frac{1}{r^2}}$ [dB]', fontsize=20)
        plt.ylabel('MSE [dB]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.title('MSE vs inverse noise variance with inaccurate SS knowledge', fontsize=32)
        plt.grid(True)         
        plt.savefig(fileName)

    def error_evolution(self,MSE_Net, trace_Net,MSE_KF, trace_KF):
        fileName = self.folderName + 'error_evolution'
        fontSize = 32
        # Figure
        fig, axs = plt.subplots(2, figsize = (25, 10))
        # x_axis
        x_plt = range(0, MSE_Net.size()[0])
        ## Figure 1: Error
        # Net
        y_plt1 = MSE_Net.detach().numpy()
        axs[0].plot(x_plt, y_plt1, '-bo', label=error_evol[0])
        y_plt2 = trace_Net.detach().numpy()
        axs[0].plot(x_plt, y_plt2, '--yo', label=error_evol[1])
        # EKF
        y_plt3 = MSE_KF.detach().numpy()
        axs[0].plot(x_plt, y_plt3, '-ro', label=error_evol[2])
        y_plt4 = trace_KF.detach().numpy()
        axs[0].plot(x_plt, y_plt4, '--go', label=error_evol[3])
        axs[0].legend(loc="upper right")

        ## Figure 2: Error Deviation
        # Net
        y_plt5 = MSE_Net.detach().numpy() - trace_Net.detach().numpy()
        axs[1].plot(x_plt, y_plt5, '-bo', label=error_evol[4])
        # EKF
        y_plt6 = MSE_KF.detach().numpy() - trace_KF.detach().numpy()
        axs[1].plot(x_plt, y_plt6, '-ro', label=error_evol[5])
        axs[1].legend(loc="upper right")
        
        axs[0].set(xlabel='Timestep', ylabel='Error [dB]')
        axs[1].set(xlabel='Timestep', ylabel='Error Deviation[dB]')
        axs[0].grid(True)
        axs[1].grid(True)
        fig.savefig(fileName)
