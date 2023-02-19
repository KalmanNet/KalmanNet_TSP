"""This file contains the settings for the simulation"""
import argparse

def general_settings():
    ### Dataset settings
        # Sizes
    parser = argparse.ArgumentParser(prog = 'KalmanNet',\
                                     description = 'Dataset, training and network parameters')
    parser.add_argument('--N_E', type=int, default=1000, metavar='trainset-size',
                        help='input training dataset size (# of sequences)')
    parser.add_argument('--N_CV', type=int, default=100, metavar='cvset-size',
                        help='input cross validation dataset size (# of sequences)')
    parser.add_argument('--N_T', type=int, default=200, metavar='testset-size',
                        help='input test dataset size (# of sequences)')
    parser.add_argument('--T', type=int, default=100, metavar='length',
                        help='input sequence length')
    parser.add_argument('--T_test', type=int, default=100, metavar='test-length',
                        help='input test sequence length')
        # Random length
    parser.add_argument('--randomLength', type=bool, default=False, metavar='rl',
                    help='if True, random sequence length')
    parser.add_argument('--T_max', type=int, default=1000, metavar='maximum-length',
                    help='if random sequence length, input max sequence length')
    parser.add_argument('--T_min', type=int, default=100, metavar='minimum-length',
                help='if random sequence length, input min sequence length')
        # Random initial state
    parser.add_argument('--randomInit_train', type=bool, default=False, metavar='ri_train',
                        help='if True, random initial state for training set')
    parser.add_argument('--randomInit_cv', type=bool, default=False, metavar='ri_cv',
                        help='if True, random initial state for cross validation set')
    parser.add_argument('--randomInit_test', type=bool, default=False, metavar='ri_test',
                        help='if True, random initial state for test set')
    parser.add_argument('--variance', type=float, default=100, metavar='variance',
                        help='input variance for the random initial state with uniform distribution')
    parser.add_argument('--distribution', type=str, default='normal', metavar='distribution',
                        help='input distribution for the random initial state (uniform/normal)')


    ### Training settings
    parser.add_argument('--use_cuda', type=bool, default=False, metavar='CUDA',
                        help='if True, use CUDA')
    parser.add_argument('--n_steps', type=int, default=1000, metavar='N_steps',
                        help='number of training steps (default: 1000)')
    parser.add_argument('--n_batch', type=int, default=20, metavar='N_B',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--CompositionLoss', type=bool, default=False, metavar='loss',
                        help='if True, use composition loss')
    parser.add_argument('--alpha', type=float, default=0.3, metavar='alpha',
                        help='input alpha [0,1] for the composition loss')

    
    ### KalmanNet settings
    parser.add_argument('--in_mult_KNet', type=int, default=5, metavar='in_mult_KNet',
                        help='input dimension multiplier for KNet')
    parser.add_argument('--out_mult_KNet', type=int, default=40, metavar='out_mult_KNet',
                        help='output dimension multiplier for KNet')

    args = parser.parse_args()
    return args
