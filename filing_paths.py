path_model_Lor = 'Simulations/Lorenz_Atractor/'
path_model_NCLT = 'Simulations/NCLT/'
path_model_Toy = 'Simulations/Toy_problems/'
path_model_Pendulum = 'Simulations/Pendulum/'

path_model = path_model_Toy
path_session = path_model + 'Experiment_GPS_RNN/'

'''
# Function for Outputting Path (used for cluster)
def getPathSession(noise_values, path_experiment = 'Testing/Experiment 1_Lower Taylor/', path_Taylor='J_gen=5/J_mdl=5/'):
    path_noise = 'r_' + noise_values[0] + '/q_e' + noise_values[1] + '/'
    print(path_noise)
    path_session = path_model + path_experiment + path_noise + path_Taylor
    return path_session
'''