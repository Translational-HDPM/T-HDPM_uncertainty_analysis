from functools import partial
import numpy as np
from joblib import Parallel, delayed

# Function that returns the true label given a patient ID
def get_true_label(patient_id, labels):
    filt = labels["Isolate ID"] == float(patient_id)
    return labels[filt]["Disease"].values[0]

# Sampling function performing the Monte Carlo simulations
def Simulation(means, std, coefficients):
    return np.sum(np.multiply(coefficients, np.random.normal(means, std, size=(1, len(coefficients)))))

# Function to perform anti-logit operation on the linear score 
def cl_score(linear_score, gamma = 0):
    temp = gamma + linear_score
    classifier_score = np.exp(temp) / (1 + np.exp(temp))
    return classifier_score

# Function to calculate subject wise mean and std of simulated scores.
# Parallelized using joblib.
def run_sim_one_patient_mean_sd(col, percent, num_runs, seed):
    np.random.seed(seed)
    std = [percent/100 * val for val in col]
    std = np.abs(std)
    if num_runs > 50:
        temp_Sim = Parallel(n_jobs=-1)(delayed(Simulation)(
            col, std, coefficients)  for _ in range(num_runs))
    else:
        temp_Sim = [Simulation(col, std, coefficients)  \
                    for _ in range(num_runs)]

    return [np.mean(temp_Sim),np.std(temp_Sim)]

# Function to calculate the classifier score for each simulation of "num_runs" simulations, corresponding to each subject.
# Parallelized using joblib.
def run_sim_one_patient(col, percent, num_runs, coefficients, seed):
    np.random.seed(seed)
    score_calc_fn = lambda x: cl_score(Simulation(col, std, coefficients))
    std = [percent/100 * val for val in col]
    std = np.abs(std)
    if num_runs > 50:
        score_vals = Parallel(n_jobs=-1)(delayed(score_calc_fn)(_)\
                            for _ in range(num_runs))
    else:
        score_vals = [score_calc_fn(_) for _ in range(num_runs)]
    temp_Sim = np.asarray(score_vals)
    return temp_Sim

# This score is the classifer linear score we want to compare with the simulated scores
def linear_score(coefficients, col):
    linear_score = np.sum(coefficients * col, axis=0)
    return linear_score

# We define a function whose input is TPM and outputs the corresponding Zscore
def z_score(x):
    return (x-x['Mean'])/x['Std']

# Function to calculate true positives, false positives, true negatives and
# false negatives given a threshold
def calculate_mispredictions(Patients_df_zScore, 
                          coefficients, labels, 
                          thresh, num_runs, uncertainty, seed):
    false_pos = 0 # counter, stores number of points on second quadrant
    false_neg = 0
    true_pos = 0
    true_neg = 0

    for i in range(243):
        scores = run_sim_one_patient(Patients_df_zScore.iloc[:, i], 
                                    uncertainty, 
                                    num_runs, coefficients, seed)
        y_0 = cl_score(linear_score(coefficients, Patients_df_zScore.iloc[:, i]))
        y_true = get_true_label(Patients_df_zScore.columns[i], labels)
        for j in range(len(scores)):
            if y_true == 1 and scores[j] < thresh:
                false_neg = false_neg + 1
            elif y_true == 0 and scores[j] > thresh:
                false_pos = false_pos + 1
            elif y_true == 1 and scores[j] > thresh:
                true_pos = true_pos + 1
            elif y_true == 0 and scores[j] < thresh:
                true_neg = true_neg + 1
 
    return true_pos, true_neg, false_pos, false_neg

# Function that calculates L2 norm of sensitivity from a target sensitivity
# given a threshold. Used as an objective function to calculate best threshold
# given a target sensitivity
def calculate_mse_sensitivity(
                threshold, Patients_df_zScore, coefficients, labels,
                num_runs, uncertainty, target_sensitivity, seed):
    true_pos, _, _, false_neg = calculate_mispredictions(
                                  Patients_df_zScore, 
                                  coefficients, labels, 
                                  threshold, num_runs, uncertainty, seed)
    sensitivity_calc = true_pos/(true_pos+false_neg)
    return (target_sensitivity-sensitivity_calc)**2

# Function that calculates L2 norm of specificity from a target specificity
# given a threshold. Used as an objective function to calculate best threshold
# given a target specificity
def calculate_mse_specificity(
                threshold, Patients_df_zScore, coefficients, labels,
                num_runs, uncertainty, target_specificity, seed):
    _, true_neg, false_pos, _ = calculate_mispredictions(
                                  Patients_df_zScore, 
                                  coefficients, labels, 
                                  threshold, num_runs, uncertainty, seed)
    specificity_calc = true_neg/(true_neg+false_pos)
    return (target_specificity-specificity_calc)**2

# Function to return the threshold that results in a specified sensitivity
# given a set of classifier coefficients, using linear search
def find_best_threshold_at_sensitivity(
                        target_sensitivity, thresh_low_bnd, thresh_up_bnd,
                        Patients_df_zScore, coefficients, labels, seed):
    sample_space = np.linspace(thresh_low_bnd, thresh_up_bnd, 500)
    objective = partial(calculate_mse_sensitivity, 
                        Patients_df_zScore=Patients_df_zScore,
                        coefficients=coefficients, labels=labels,
                        num_runs=1, uncertainty=0, 
                        target_sensitivity=target_sensitivity, seed=seed)
    mse_vals = Parallel(n_jobs=-1)(delayed(objective)(thresh) \
                    for thresh in sample_space)
    mse = np.asarray(mse_vals, dtype=np.float32)
    return np.round(sample_space[np.argmin(mse)], 4)

# Function to return the threshold that results in a specified specificity
# given a set of classifier coefficients, using linear search
def find_best_threshold_at_specificity(
                        target_specificity, thresh_low_bnd, thresh_up_bnd,
                        Patients_df_zScore, coefficients, labels, seed):
    sample_space = np.linspace(thresh_low_bnd, thresh_up_bnd, 500)
    objective = partial(calculate_mse_specificity, 
                        Patients_df_zScore=Patients_df_zScore,
                        coefficients=coefficients, labels=labels,
                        num_runs=1, uncertainty=0, 
                        target_specificity=target_specificity, seed=seed)
    mse_vals = Parallel(n_jobs=-1)(delayed(objective)(thresh) \
                    for thresh in sample_space)
    mse = np.asarray(mse_vals, dtype=np.float32)
    return np.round(sample_space[np.argmin(mse)], 4)
