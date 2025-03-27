"""
Plotting functions for Monte Carlo simulations
"""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.classifier import (antilogit_classifier_score,
                            linear_classifier_score,
                            sample_single_patient)

sns.set_style("ticks")


def plot_uncertainty_at_threshold(
    *,
    z_scores_df: pd.DataFrame,
    coefficients: Sequence[float],
    num_runs: int = 100,
    uncertainty: int = 20,
    thresh: float = 0.5,
    num_patients: int = 243,
) -> tuple[int, int, int]:
    """
    Inputs:

        %RSD (uncertainty): Range of relative standard deviation values.
        Threshold (thresh): Threshold value for classification.
        Number of Monte Carlo Simulations (num_runs).

    Outputs:

        Scatter plot displaying simulation classifier scores against subject scores.
        Classification of outcomes as False Positives (FP), False Negatives (FN), 
        True Positives (TP), and True Negatives (TN).

    Pseudocode:

      1. Initialize Iteration over %RSD:
            For each value in the %RSD range, begin processing.

      2. Compute Subject Scores:
            For each subject:
            a. Linear Score Calculation: Compute linear scores by multiplying the z-scores 
            (from z_scores_df) by a coefficient.
                Function: linear_classifier_score.
                b. Classifier Score Calculation: Convert linear scores to classifier scores
                    using an anti-logit operation.
                Function: antilogit_classifier_score.

      3. Compute Simulation Scores:
            For each subject, generate num_runs simulation scores using the classifier model.
                Function: sample_single_patient.

      4. Prepare Data for Plotting:
        a. Create Matching Arrays for Subject and Simulation Scores:
            x_data: Array of subject scores in the same shape as the simulation scores.
            b. Store Simulation Scores for Simplicity:
            y_data: Array containing simulation scores.

      5. Classify Simulation Outcomes:
            For each simulation score for each subject (iterate num_runs times):
                Use the threshold to classify outcomes into FP, FN, TP, and TN using 
                conditional statements.

      6. Calculate Accuracy Metrics:
            Based on the FP, FN, TP, and TN classifications, calculate accuracy measures for
            each subject.

      7. Repeat Process for All Subjects:
            Repeat steps 2–6 for each subject.

      8. Generate Scatter Plot:
            Plot simulation scores (y-axis) against subject scores (x-axis).
            Add two perpendicular threshold lines on the plot (one on the x-axis and 
            one on the y-axis) to categorize scatter dots into FP, FN, TP, and TN.
    """

    false_pos = false_neg = 0
    # keeps track of subjects whose score is unreliable under the assumed variation
    num_subj_unreliable = 0

    plt.figure(figsize=(10, 10))
    for i in range(num_patients):
        col = z_scores_df.iloc[:, i]
        y_data = sample_single_patient(col, coefficients, num_runs, uncertainty)
        y_0 = antilogit_classifier_score(linear_classifier_score(coefficients, col))
        x_data = np.ones_like(y_data) * y_0
        colour = np.zeros_like(x_data)
        for j in range(x_data.shape[0]):
            if x_data[j] > thresh:
                if y_data[j] < thresh:
                    colour[j] = 1
                    false_neg += 1
            elif y_data[j] > thresh:
                colour[j] = 2
                false_pos += 1
        if false_neg > 0 or false_pos > 0:
            num_subj_unreliable += 1
        plt.scatter(x_data, y_data, c=colour, cmap="Dark2_r", alpha=0.15, s=100)
    plt.axvline(x=thresh, color="g", linestyle="--")
    plt.axhline(y=thresh, color="g", linestyle="--")
    plt.xlabel("Classifier score", fontsize=24)
    plt.ylabel("Simulated scores", fontsize=24)
    plt.title("Uncertainty around threshold", fontsize=28)
    plt.text(0.4, 0.1, "TN", fontsize=20)
    plt.text(0.97, 0.1, "FN", fontsize=20)
    plt.text(0.4, 0.9, "FP", fontsize=20)
    plt.text(0.97, 0.9, "TP", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.show()

    plt.savefig(f"uncert_around_thresh_{uncertainty}pc.png")

    return false_pos, false_neg, num_subj_unreliable


def plot_v_plot(
    z_scores_df: pd.DataFrame,
    coefficients: Sequence[float],
    uncertainty_range: list[int],
    thresh: float = 0.5,
    num_runs: int = 100,
    num_patients: int = 243,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inputs:

        %RSD (uncertainty_range): Range of relative standard deviation values.
        Threshold (thresh): Classification threshold.
        Number of Monte Carlo Simulations (num_runs).

    Outputs:

        Subject accuracy based on the number of simulations, stored in a DataFrame: 
        accuracy_df. True Positives (TP), True Negatives (TN), False Positives (FP), 
        and False Negatives (FN) classifications, stored in a DataFrame: false_pos_df.

    Pseudocode:

     1. Iterate over %RSD Values:
            Begin processing with the first %RSD in the range.

     2. Calculate Subject Scores:
            For each subject:
            a. Linear Score Calculation: Compute linear scores by multiplying `z_scores_df` 
            (a DataFrame of z-scores) by a coefficient.
                Function: linear_classifier_score.
            b. Classifier Score Calculation: Convert linear scores into classifier scores 
            using an anti-logit operation.
                Function: antilogit_classifier_score.

     3. Generate Simulation Scores:
            For each subject, simulate num_runs scores corresponding to the classifier model.
                Function: run_sim_one_patient.

     4. Prepare Data for Plotting:
        a. Create Matching Subject and Simulation Score Arrays:
            x_data: Ensure the subject score array matches the shape of the simulation scores array.
            b. Store Simulation Scores:
            y_data: Simplified array containing simulation scores.

     5. Classify Simulations for Each Subject:
            Iterate over the num_runs simulation scores for each subject:
                Use conditional statements based on the threshold to classify each
                simulation score as:
                    True Positive (TP)
                    True Negative (TN)
                    False Positive (FP)
                    False Negative (FN)

     6. Calculate Accuracy for Each Subject:
            Derive accuracy metrics for the subject based on FP, FN, TP, and TN classifications.

     7. Repeat Steps 2-6 for Each Subject:
            Complete the computations for all subjects.

     8. Repeat Steps 2-7 for Each %RSD Input:
            Process each value in the %RSD range.
    """
    real_score = np.zeros(num_patients)
    accuracy = np.zeros(num_patients)

    false_pos_df = pd.DataFrame(
        columns=uncertainty_range,
        index=[
            "simulations in agreement with subject",
            "simulations in disagreement with subject",
            "Unreliable Subjects",
        ],
    )
    # keeps track of subjects whose score is unreliable under the assumed variation
    # (0 is fine, 1 is unreliable)
    num_subj_unreliable = 0
    accuracy_df = pd.DataFrame(columns=uncertainty_range)

    for i, uncertainty in enumerate(uncertainty_range):
        tot_false_pos = tot_false_neg = tot_true_pos = tot_true_neg = 0

        for j in range(num_patients):
            col = z_scores_df.iloc[:, j]
            y_data = sample_single_patient(
                col,
                coefficients,
                num_runs,
                uncertainty,
            )
            y_0 = antilogit_classifier_score(linear_classifier_score(coefficients, col))
            x_data = np.ones_like(y_data) * y_0
            false_neg = false_pos = 0
            for k in range(x_data.shape[0]):
                if x_data[k] >= thresh:
                    if y_data[k] < thresh:
                        false_neg += 1
                    else:
                        tot_true_pos += 1
                else:
                    if y_data[k] >= thresh:
                        false_pos += 1
                    else:
                        tot_true_neg += 1
            tot_false_pos += false_pos
            tot_false_neg += false_neg

            accuracy[j] = (num_runs - (false_neg) - (false_pos)) / num_runs
            real_score[j] = y_0
            if false_neg > 0 or false_pos > 0:
                num_subj_unreliable += 1

        unreliable_subjects = str(num_subj_unreliable)

        disagreement = tot_false_neg + tot_false_pos
        agreement = tot_true_neg + tot_true_pos

        accuracy_df[accuracy_df.columns[i]] = pd.Series(accuracy) * 100
        false_pos_df[false_pos_df.columns[i]] = [
            agreement,
            disagreement,
            unreliable_subjects,
        ]
        accuracy_df["Classifier Score"] = pd.Series(real_score)

        dfm = accuracy_df.melt(
            "Classifier Score", var_name="%RSD", value_name="Agreement"
        )
    sns.relplot(
        x="Classifier Score",
        y="Agreement",
        hue="%RSD",
        data=dfm,
        kind="line",
        linewidth=1,
        palette="rainbow",
    )
    patient_id = list(z_scores_df.columns.values)
    patient_id = patient_id[:num_patients]
    accuracy_df["Patient ID"] = patient_id
    accuracy_df = accuracy_df.set_index("Patient ID")

    ad = np.sum(accuracy_df["Classifier Score"] > thresh)
    nci = len(accuracy_df["Classifier Score"]) - ad
    print(f"{ad=} {nci=}")
    return accuracy_df, false_pos_df
