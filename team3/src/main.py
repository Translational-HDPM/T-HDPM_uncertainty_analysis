import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from threshold import (plot_uncert_at_thresh, sub_accuracy, 
                       sub_accuracy_smoothed_plot)
from classifier import (z_score, calculate_mispredictions, 
                        find_best_threshold_at_sensitivity, 
                        find_best_threshold_at_specificity)

NUM_RUNS = 1000      # Number of simulation runs
UNCERTAINTY = 45    # Percentage of simulated relative standard deviation
UNCERT_RANGE = [10, 25, 50]     # Percentage RSD for V-plots
THRESH = 0.86       # Classification threshold
SEED = 42          # Random number generator seed
TARGET_SENSITIVITY = 0.85       # Specified sensitivity to find best threshold
TARGET_SPECIFICITY = 0.97       # Specified specificity to find best threshold

def main():
    np.random.seed(SEED)
    
    # Reading in TPM data and true labels
    myDF = pd.read_excel(
                "../../../molecular_stethoscope/data-s23/ClusterMarkers_1819ADcohort.congregated_DR.xlsx", 
                sheet_name = 1)
    labels = pd.read_excel(
                "../../../molecular_stethoscope/data-s23/ClusterMarkers_1819ADcohort.congregated_DR.xlsx", 
                sheet_name = 0)
    labels = labels[~labels["Isolate ID"].isnull()]

    # Replacing "AD" with 1 and other labels with 0
    labels["Disease"] = labels["Disease"].apply(lambda x: 1 if x == "AD" else 0)

    # setting index row name to the gene id
    myDF = myDF.set_index('gene_id')

    #Filtering out rows: discarding the ERCC rows, ERCC is a control protocol for validation of RNA sequencing
    Patients_df = myDF[~myDF.loc[:,'Coeff'].isnull()]

    # We store the coefficients(betas) of the linear classifier in an array.
    coefficients = np.nan_to_num(np.array(Patients_df.loc[:, "Coeff"]))

    # Filtering out columns with patient data
    Patients_df = Patients_df.filter(regex='^\d+')

    # group columns by patient id
    grouped_cols = Patients_df.columns.str.split('-').str[0]

    # group columns by patient id and r1/r2 suffixes
    grouped = Patients_df.groupby(grouped_cols, axis=1)

    # apply the mean function to the r1 and r2 columns for each group
    Patients_df = grouped.apply(lambda x: x.mean(axis=1)).reset_index(drop=True)
    
    # removing patient IDs from labels dataframe that are not present in TPM data
    missing = [col for col in labels["Isolate ID"].values\
                    if col not in Patients_df.columns.values.astype(np.float64)]
    for id in labels["Isolate ID"]:
        if id not in missing:
            continue
        labels = labels[labels["Isolate ID"] != id]

    # taking mean of the replicates for subjects with multiple replicates
    Patients_df['Mean']= Patients_df.mean(axis=1)
    Patients_df['Std'] = Patients_df.iloc[:,:-1].std(axis=1)

    # Computing and storing zscores
    Patients_df_zScore = Patients_df.apply(lambda x: z_score(x), axis=1)
    
    print("---------------------------------------")
    # Sensitivity and specificity variation with threshold
    thresholds = np.linspace(0, 1, 1000)
    sensitivities = np.zeros_like(thresholds)
    specificities = np.zeros_like(thresholds)
    for count, threshold in enumerate(thresholds):
        true_pos, true_neg, false_pos, false_neg = \
                calculate_mispredictions(Patients_df_zScore, coefficients,
                                         labels, threshold, 1, 0, SEED)
        sensitivities[count] = true_pos/(true_pos+false_neg)
        specificities[count] = true_neg/(false_pos+true_neg)
    
    fig = plt.figure(figsize=(10, 10))
    plt.plot(thresholds, sensitivities, label="Sensitivity", color="k")
    plt.plot(thresholds, specificities, label="Specificity", color="r")
    plt.tick_params(labelsize=18)
    plt.xlabel("Threshold", fontsize=24)
    plt.ylabel("Measure of misprediction", fontsize=24)
    plt.legend(loc="best", fontsize=18)
    fig.savefig("Sensitivity_vs_threshold.png")
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    plt.plot(1-specificities, sensitivities)
    plt.tick_params(labelsize=18)
    plt.xlabel("1-specificity", fontsize=24)
    plt.ylabel("Sensitivity", fontsize=24)
    fig.savefig("ROC.png")
    plt.close()

    print("---------------------------------------")
    # Scatter plot of scores of simulated points versus scores of original points
    print(f"Threshold = {THRESH}")
    true_pos, true_neg, false_pos, false_neg, _ = \
           plot_uncert_at_thresh(Patients_df_zScore, coefficients, labels, 
                        NUM_RUNS, UNCERTAINTY, THRESH, SEED)
    print(f"TP = {true_pos}, TN = {true_neg}, FP = {false_pos}, FN = {false_neg}")
    print(f"Actual sensitivity = {true_pos/(true_pos+false_neg):.4f}, ", end="")
    print(f"Actual specificity = {true_neg/(true_neg+false_pos):.4f}")
 
    # V plot of agreement between simulated scores and original scores,
    # with smoothing
    sub_accuracy_smoothed_plot(Patients_df_zScore, coefficients,
                UNCERT_RANGE, THRESH, NUM_RUNS, SEED)
    print("---------------------------------------")

    # plots for threshold given a target sensitivity
    thresh = find_best_threshold_at_sensitivity(TARGET_SENSITIVITY, 0.01, 0.99,
                                Patients_df_zScore, coefficients, labels, SEED)
    print(f"Target sensitivity = {TARGET_SENSITIVITY}, threshold = {thresh}")

    for uncert in UNCERT_RANGE:
        true_pos, true_neg, false_pos, false_neg, _ = plot_uncert_at_thresh(
                            Patients_df_zScore, coefficients, labels, 
                            NUM_RUNS, uncert, thresh, SEED)
        print(f"TP = {true_pos}, TN = {true_neg}, FP = {false_pos}, FN = {false_neg}")
        print(f"Actual sensitivity = {true_pos/(true_pos+false_neg):.4f}, ", end="")
        print(f"Actual specificity = {true_neg/(true_neg+false_pos):.4f}")
    sub_accuracy_smoothed_plot(Patients_df_zScore, coefficients,
                UNCERT_RANGE, thresh, NUM_RUNS, SEED)
    print("---------------------------------------")

    # plots for threshold given a target specificity
    thresh = find_best_threshold_at_specificity(TARGET_SPECIFICITY, 0.01, 0.99,
                                Patients_df_zScore, coefficients, labels, SEED)
    print(f"Specificity = {TARGET_SPECIFICITY}, threshold = {thresh}")
    for uncert in UNCERT_RANGE:
        true_pos, true_neg, false_pos, false_neg, _ = plot_uncert_at_thresh(
                            Patients_df_zScore, coefficients, labels, 
                            NUM_RUNS, uncert, thresh, SEED)
        print(f"TP = {true_pos}, TN = {true_neg}, FP = {false_pos}, FN = {false_neg}")
        print(f"Actual sensitivity = {true_pos/(true_pos+false_neg):.4f}, ", end="")
        print(f"Actual specificity = {true_neg/(true_neg+false_pos):.4f}")
    sub_accuracy_smoothed_plot(Patients_df_zScore, coefficients,
                UNCERT_RANGE, thresh, NUM_RUNS, SEED)
    print("---------------------------------------")

if __name__ == "__main__":
    main()
