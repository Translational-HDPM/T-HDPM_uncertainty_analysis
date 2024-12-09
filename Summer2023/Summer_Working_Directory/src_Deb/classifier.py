import numpy as np

np.random.seed(46215423)

# Sampling function performing the Monte Carlo simulations
def Simulation(means, std, coefficients):
    return np.sum(np.multiply(coefficients, np.random.normal(means, std, size=(1, len(coefficients)))))

# Function to perform anti-logit operation on the linear score 
def cl_score(linear_score, gamma = 0):
    temp = gamma + linear_score
    classifier_score = np.exp(temp) / (1 + np.exp(temp))
    return classifier_score

# Function to calculate subject wise mean and std of simulated scores 
def run_sim_one_patient_mean_sd(col,percent):
    std = [percent/100 * val for val in col]
    std = np.abs(std)
    temp_Sim = [Simulation(col, std, coefficients) for _ in range(numRuns)]
    return [np.mean(temp_Sim),np.std(temp_Sim)]

# Function to calculate the classifier score for each simulation of "num_runs" simulations, corresponding to each subject
def run_sim_one_patient(col, percent, num_runs, coefficients):
    std = [percent/100 * val for val in col]
    std = np.abs(std)
    temp_Sim = np.asarray([cl_score(Simulation(col, std, coefficients)) for _ in range(num_runs)])
    return temp_Sim

# This score is the classifer linear score we want to compare with the simulated scores
def linear_score(coefficients, col):
    linear_score = np.sum(coefficients * col, axis=0)
    return linear_score

# We define a function whose input is TPM and outputs the corresponding Zscore
def z_score(x):
    return (x-x['Mean'])/x['Std']
