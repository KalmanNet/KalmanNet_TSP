# KalmanNet

## Link to paper

[KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics](https://arxiv.org/abs/2107.10043)

## Running code

### Branch - main

This branch simulates architecture #1 in our paper. There are two main files simulating the linear and non-linear cases respectively.

* Linear case

```
python3 main_linear.py
```

* Non-linear Lorenz Attractor(LA) case

```
python3 main_lorenz.py
```

* Non-linear pendulum, toy problem or real-world NCLT case

change the path_model parameter in filling_paths.py accordingly and then run main_lorenz.py.


### Branch - new_architecture

This branch simulates architecture #2 in our paper. There are two main files simulating the Discrete-Time and Decimation cases respectively.

* Discrete-Time LA case

```
python3 main_lor_DT.py
```

* Decimated LA case

```
python3 main_lor_decimation.py
```

For pendulum, toy problem or real-world NCLT case, similarly, you could change the path_model parameter in filling_paths.py.


## Introduction to other files

### Branch - main

* KNet/

This folder is used to store the trained KalmanNet model and pipeline results of your simulation.

* Simulations/

This folder stores the dataset for synthetic linear case, as well as model and parameter files for LA, pendulum, toy problem and real-world NCLT.

* EKF.py & EKF_test.py

This is where we define Extended Kalman Filter (EKF) model and its testing.

* Extended_data.py

This is a parameter-setting and data-generating/loading/preprocessing file.

You could set the number of Training/Cross-Validation(CV)/Testing examples through N_E/N_CV/N_T.

You could set trajectory length of Training/CV examples for linear case through T, while T_test is for Testing trajectory length.

You could set the synthetic linear model through F10 and H10, and uncomment 2x2, 5x5, 10x10 according to your needs.

* KalmanNet_nn.py & Extended_KalmanNet_nn.py

These files describe the architecture #1 of KalmanNet model for linear and non-linear cases respectively.

* Linear_sysmdl.py & Extended_sysmdl.py

These are system model files for linear and non-linear cases respectively. They store system information (F/f, H/h, Q, R, m, n) and define functions for generating data according to your system model.

* filling_paths.py

This is where you switch between different system models.

* Linear_KF.py & KalmanFilter_test.py

This is where we define Linear Kalman Filter (KF) model and its testing.

* Optimal_q_search.py

This is a main file to search for optimal Q or R for fair comparison with KF.

* PF_test.py & UKF_test.py

These files defines the testing of Particle Filter (PF) and Unscented Kalman Filter (UKF) benchmarks.

* Pipeline_KF.py & Pipeline_EKF.py

These are the pipeline files for linear and non-linear cases of KalmanNet respectively. The pipeline mainly defines the Training/CV/Testing processes of KalmanNet.

* Plot.py

This file mainly defines the plotting of training process, histogram, MSE results, etc.

### Branch - new_architecture

Since the file arrangement is very simular with the main branch, here we only briefly introduce the different parts.

* KNet/KNet_TSP

This folder stores the simulation results in our paper.

* KalmanNet_nn.py

This file specifies KalmanNet architecture #2.





