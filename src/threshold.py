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