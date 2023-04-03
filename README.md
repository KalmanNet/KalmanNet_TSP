# KalmanNet

## This branch is the old version. Please refer to branch main for better performance.


## Link to paper

[KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics](https://arxiv.org/abs/2107.10043)

## Running code

### Branch - architecture #2

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

### Branch - architecture #2

Since the file arrangement is very simular with the main branch, here we only briefly introduce the different parts.

* KNet/KNet_TSP

This folder stores the simulation results in our paper.

* KalmanNet_nn.py

This file specifies KalmanNet architecture #2.





