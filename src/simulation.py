"""
Functions and classes for Monte Carlo simulations on RNA-seq data for Alzheimer's disease.
"""

import warnings
from typing import Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import scipy.stats as st
from joblib import Parallel, delayed
from .logreg_classifier import (
    linear_classifier_score,
    linear_classifier_subscores,
    antilogit_classifier_score,
    z_score,
)
from .dtypes import NumpyFloat32Array1D

warnings.filterwarnings("ignore")  # ignore all warnings


@dataclass
class SingleUncertaintyResults:
    """
    Dataclass storing simulation results for classification for a single uncertainty level.

    Parameters
    ----------
    gt_series
        A Pandas series containing predictions from a classifier from original TPM
        values of subjects (no added noise). The labels are integers. Each patient
        ID has the value of a Numpy array of integers of shape (number of samples,).
    preds_series
        A Pandas series containing predictions from a classifier from simulated TPM
        values of subjects. The labels are integers. Each patient ID has the value of
        a Numpy array of integers of shape (number of samples,).
    gt_labels
        A Pandas series containing predictions from a classifier from original TPM
        values of subjects (no added noise). The labels are integers. Each patient
        ID has the value of an integer corresponding to the "ground truth" label.
    pred_labels
        A Pandas series containing predictions from a classifier from simulated TPM
        values of subjects. The labels are integers. Each patient ID has the value of
        an integer corresponding to the predicted label (accounting for the criterion
        for differential classification).
    """

    gt_series: pd.Series = field(default_factory=pd.Series)
    preds_series: pd.Series = field(default_factory=pd.Series)
    gt_labels: pd.Series = field(default_factory=pd.Series)
    pred_labels: pd.Series = field(default_factory=pd.Series)


class MultiUncertaintyResults:
    """
    Class to store simulation results from multiple simulation runs with single
    and two threshold simulation scenarios.

    Parameters
    ----------
    score_stats
        A Pandas dataframe containing summary statistics for classifier outputs,
        including probability values and linear classifier values.
    pos_subscore_arrs
        A dictionary containing Numpy arrays of positive subscores (dot product of
        positive classifier coefficients and z-score values) for all uncertainty
        levels.
    neg_subscore_arrs
        A dictionary containing Numpy arrays of positive subscores (dot product of
        negative classifier coefficients and z-score values) for all uncertainty
        levels.
    lin_score_arrs
        A dictionary containing Numpy arrays of linear classifier scores (dot
        product of classifier coefficients and z-score values) for all
        uncertainty levels.
    pred_prob_arrs
        A dictionary containing Numpy arrays of probability scores (classifier
        outputs) for all uncertainty levels.
    single_thres_gt_series
        A dictionary containing predictions from a classifier from original TPM
        values of subjects (no added noise) for each uncertainty level (contents
        of the `gt_series` attribute for `SingleThresholdResults` instances)
    single_thres_pred_series
        A dictionary containing contents of the `pred_series` attribute for
        `SingleThresholdResults` instances for all uncertainty levels.
    dual_thres_gt_series
        A dictionary containing contents of the `gt_series` attribute for
        `DualThresholdResults` instances for all uncertainty levels.
    dual_thres_pred_series
        A dictionary containing contents of the `pred_series` attribute for
        `DualThresholdResults` instances for all uncertainty levels.
    single_thres_gt_labels
        A pandas Series containing contents of the `gt_labels` attribute for
        `SingleThresholdResults` instances.
    single_thres_pred_labels
        A dictionary containing contents of the `pred_labels` attribute for
        `SingleThresholdResults` instances for all uncertainty levels.
    dual_thres_gt_labels
        A pandas Series containing contents of the `gt_labels` attribute for
        `DualThresholdResults` instances.
    dual_thres_pred_labels
        A dictionary containing contents of the `pred_labels` attribute for
        `DualThresholdResults` instances for all uncertainty levels.
    """

    def __init__(self, uncertainties: list[int | float]):
        self.score_stats: pd.DataFrame = pd.DataFrame(
            index=uncertainties,
            columns=["mean_lin", "std_lin", "mean_probs", "std_probs"],
        )

        self.pos_subscore_arrs: dict[int, NumpyFloat32Array1D] = {
            uncert: None for uncert in uncertainties
        }
        self.neg_subscore_arrs: dict[int, NumpyFloat32Array1D] = {
            uncert: None for uncert in uncertainties
        }
        self.lin_score_arrs: dict[int, NumpyFloat32Array1D] = {
            uncert: None for uncert in uncertainties
        }
        self.pred_prob_arrs: dict[int, NumpyFloat32Array1D] = {
            uncert: None for uncert in uncertainties
        }
        self.single_thres_gt_series: dict[int, pd.Series] = {
            uncert: None for uncert in uncertainties
        }
        self.single_thres_pred_series: dict[int, pd.Series] = {
            uncert: None for uncert in uncertainties
        }
        self.dual_thres_gt_series: dict[int, pd.Series] = {
            uncert: None for uncert in uncertainties
        }
        self.dual_thres_pred_series: dict[int, pd.Series] = {
            uncert: None for uncert in uncertainties
        }
        self.single_thres_gt_labels: pd.Series = None
        self.single_thres_pred_labels: dict[int, pd.Series] = {
            uncert: None for uncert in uncertainties
        }
        self.dual_thres_gt_labels: pd.Series = None
        self.dual_thres_pred_labels: dict[int, pd.Series] = {
            uncert: None for uncert in uncertainties
        }


def simulate_sampling_experiment(
    tpm_df: pd.DataFrame,
    sampler: Callable[[float, float, int], NumpyFloat32Array1D],
    *,
    dual_thres_1: float,
    dual_thres_2: float,
    single_thres: float,
    diff_class_lim: int,
    uncertainty: float,
    n_samples: int,
    coefficients: np.ndarray | pd.Series,
    seed: int | np.random.SeedSequence | None = None,
    worker_id: int = 1,
) -> tuple[
    SingleUncertaintyResults,
    SingleUncertaintyResults,
    NumpyFloat32Array1D,
    NumpyFloat32Array1D,
    NumpyFloat32Array1D,
    NumpyFloat32Array1D,
]:
    """
    Perform simulation experiments for single and dual thresholds, return the
    predicted and ground truth classes and number of differentially classified
    individuals for each class, along with all predicted linear scores and
    probabilities.

    Parameters
    ----------
    tpm_df
        A Pandas dataframe containing TPM values for patients that will be used
        to calculate predictions using a logistic regression classifier.
    sampler
        A Python function that accepts a mean value, a relative standard deviation
        (RSD) value, number of points and a random number seed argument and outputs
        a NumPy array containing simulated Monte Carlo samples.
    dual_thres_1
        The lower threshold value for two threshold classification.
    dual_thres_2
        The upper threshold value for two threshold classification.
    single_thres
        The single threshold value for binary (single threshold) classification.
    diff_class_lim
        Number of simulated scores that must be different from the "ground truth"
        scores for a subject to be considered differentially classified.
    uncertainty
        Percentage noise (relative standard deviation) to simulate.
    n_samples
        Number of Monte Carlo simulations to generate for each TPM value.
    coefficients
        Coefficients of the logistic regression classifier.
    seed
        Seed for random number generation. Defaults to None.
    worker_id
        An integer ID representing the ID of the process if using parallel processing.
        Defaults to 1.

    Returns
    -------
    tuple[
        SingleUncertaintyResults,
        SingleUncertaintyResults,
        NumpyFloat32Array1D,
        NumpyFloat32Array1D,
        NumpyFloat32Array1D,
        NumpyFloat32Array1D,
    ]
        Two `SingleUncertaintyResults` instances, each representing the results
        of the single and dual threshold simulations respec- tively, and 1D
        NumPy arrays representing negative subscores, positive subscores, linear
        classifier scores and probability scores respectively.

    Raises
    ------
    ValueError
        1. When the values of any of the dual thresholds or the single threshold are not
            between 0 and 1.
        2. When the lower threshold exceeds the upper threshold in the dual threshold
            scenario.
        3. When the differential classification limit specified is not between 1 and the
            maximum number of simulated samples.
    """
    if not 0 <= dual_thres_1 <= 1 and 0 <= dual_thres_2 <= 1 and 0 <= single_thres <= 1:
        raise ValueError("Thresholds should be between 0 and 1.")
    if dual_thres_1 >= dual_thres_2:
        raise ValueError("dual_thres_1 must be less than dual_thres_2.")
    if not 1 <= diff_class_lim <= n_samples:
        raise ValueError(
            "The limit for differential classification should be"
            f" between 1 and {n_samples}."
        )

    pred_linear_scores = []
    pred_probs = []
    pred_pos_subscores, pred_neg_subscores = [], []

    n_features, num_patients = tpm_df.shape

    # Single and dual threshold result variables
    single_thres_res, dual_thres_res = (
        SingleUncertaintyResults(),
        SingleUncertaintyResults(),
    )

    _means, _stds = tpm_df.mean(axis=1), tpm_df.std(axis=1)

    # Generate random number seed sequence for seeds for sampler
    seed_seq = np.random.SeedSequence([worker_id, seed])
    for j in range(num_patients):
        samples = np.zeros((n_features, n_samples))
        patient_id = tpm_df.columns[j]
        
        # Spawn n_feature seeds, one seed per feature
        seeds = seed_seq.spawn(n_features)
        for i in range(n_features):
            mean = tpm_df.iloc[i, j]

            # Generate Monte Carlo samples
            samples[i] = sampler(mean, uncertainty / 100, n_samples, seeds[i])

        # Convert sampled TPMs to z-scores
        samples = z_score(
            samples, _means.values.reshape(-1, 1), _stds.values.reshape(-1, 1)
        )

        # Get predicted probabilities from z-scores
        neg_subscores, pos_subscores = linear_classifier_subscores(
            coefficients, samples
        )
        lin_scores = pos_subscores + neg_subscores
        probs = antilogit_classifier_score(lin_scores)

        # Keep track of linear scores, positive and negative subscores and simulated probabilities
        pred_linear_scores.append(lin_scores)
        pred_probs.append(probs)
        pred_pos_subscores.append(pos_subscores)
        pred_neg_subscores.append(neg_subscores)

        # Get ground truth probabilities from original TPMs
        tpm_z_scores = z_score(tpm_df.iloc[:, j], _means, _stds)
        y_0 = antilogit_classifier_score(
            linear_classifier_score(coefficients, tpm_z_scores)
        )

        # Single threshold calculations
        single_thres_res.gt_labels.loc[patient_id] = 0 if y_0 < single_thres else 1

        preds = np.zeros(n_samples, dtype=np.int64)
        preds[(single_thres <= probs)] = 1

        if (
            np.sum(preds != single_thres_res.gt_labels.loc[patient_id])
            >= diff_class_lim
        ):
            single_thres_res.pred_labels.loc[patient_id] = (
                1 if single_thres_res.gt_labels.loc[patient_id] == 0 else 0
            )
        else:
            single_thres_res.pred_labels.loc[patient_id] = (
                single_thres_res.gt_labels.loc[patient_id]
            )

        gt_arr = (
            np.ones(n_samples, dtype=np.int64)
            * single_thres_res.gt_labels.loc[patient_id]
        )

        single_thres_res.gt_series.loc[patient_id] = gt_arr
        single_thres_res.preds_series.loc[patient_id] = preds

        # Dual threshold calculations
        if y_0 < dual_thres_1:
            dual_thres_res.gt_labels.loc[patient_id] = 0
        elif dual_thres_1 <= y_0 < dual_thres_2:
            dual_thres_res.gt_labels.loc[patient_id] = 1
        else:
            dual_thres_res.gt_labels.loc[patient_id] = 2

        preds = np.zeros(n_samples, dtype=np.int64)
        preds[(dual_thres_1 <= probs) & (probs < dual_thres_2)] = 1
        preds[probs >= dual_thres_2] = 2

        if np.sum(preds != dual_thres_res.gt_labels.loc[patient_id]) >= diff_class_lim:
            dual_thres_res.pred_labels.loc[patient_id] = st.mode(
                preds[preds != dual_thres_res.gt_labels.loc[patient_id]]
            )[0]
        else:
            dual_thres_res.pred_labels.loc[patient_id] = dual_thres_res.gt_labels.loc[
                patient_id
            ]

        gt_arr = (
            np.ones(n_samples, dtype=np.int64)
            * dual_thres_res.gt_labels.loc[patient_id]
        )

        dual_thres_res.gt_series.loc[patient_id] = gt_arr
        dual_thres_res.preds_series.loc[patient_id] = preds

    lin_scores_all = np.hstack(pred_linear_scores)
    pred_probs_all = np.hstack(pred_probs)

    pos_subscores_all = np.hstack(pred_pos_subscores)
    neg_subscores_all = np.hstack(pred_neg_subscores)
    return (
        single_thres_res,
        dual_thres_res,
        neg_subscores_all,
        pos_subscores_all,
        lin_scores_all,
        pred_probs_all,
    )


def simulate_multiple_uncertainties(
    tpm_df: pd.DataFrame,
    sampler: Callable[[float, float, int], NumpyFloat32Array1D],
    uncertainties: list[int | float],
    *,
    thres_low: float,
    thres_high: float,
    single_thres: float,
    coefficients: pd.Series,
    diff_class_lim: int,
    n_samples: int = 1000,
    num_workers: int = 1,
    seed: int | None = None,
) -> MultiUncertaintyResults:
    """
    Run simulation of MC sampling for multiple values of uncertainties for a
    given dataset and criteria.

    Parameters
    ----------
    tpm_df
        A Pandas dataframe containing TPM values for patients that will be used
        to calculate predictions using a logistic regression classifier.
    sampler
        A Python function that accepts a mean value, a relative standard deviation
        (RSD) value, number of points and a random number seed argument and outputs
        a NumPy array containing simulated Monte Carlo samples.
    uncertainties
        A list of percentages of uncertainties to simulate.
    thres_low
        The lower threshold value for two threshold classification.
    thres_high
        The upper threshold value for two threshold classification.
    single_thres
        The single threshold value for binary (single threshold) classification.
    coefficients
        Coefficients of the logistic regression classifier.
    diff_class_lim
        Number of simulated scores that must be different from the "ground truth"
        scores for a subject to be considered differentially classified.
    n_samples
        Number of Monte Carlo simulations to generate for each TPM value.
    num_workers
        Number of parallel jobs to run for simulation. Defaults to 1. Should
        not exceed the number of processors (CPU cores) in the machine.
    seed
        Seed for random sampling. Defaults to None.

    Returns
    -------
    MultiUncertaintyResults
        An instance of `MultiUncertaintyResults` containing output of the conducted
        simulations.
    """
    res = MultiUncertaintyResults(uncertainties)
    outputs = Parallel(n_jobs=num_workers)(
        delayed(simulate_sampling_experiment)(
            tpm_df,
            sampler,
            dual_thres_1=thres_low,
            dual_thres_2=thres_high,
            single_thres=single_thres,
            coefficients=coefficients,
            uncertainty=uncertainty,
            diff_class_lim=diff_class_lim,
            n_samples=n_samples,
            seed=seed,
            worker_id=worker_id,
        )
        for worker_id, uncertainty in enumerate(uncertainties)
    )
    for uncertainty, output in zip(uncertainties, outputs):
        (
            single_thres_res,
            dual_thres_res,
            neg_subscores,
            pos_subscores,
            lin_scores,
            pred_probs,
        ) = output
        (
            res.single_thres_gt_series[uncertainty],
            res.single_thres_pred_series[uncertainty],
        ) = single_thres_res.gt_series, single_thres_res.preds_series
        (
            res.dual_thres_gt_series[uncertainty],
            res.dual_thres_pred_series[uncertainty],
        ) = dual_thres_res.gt_series, dual_thres_res.preds_series
        res.single_thres_pred_labels[uncertainty] = single_thres_res.pred_labels
        res.dual_thres_pred_labels[uncertainty] = dual_thres_res.pred_labels
        if res.single_thres_gt_labels is None:
            res.single_thres_gt_labels = single_thres_res.gt_labels
        if res.dual_thres_gt_labels is None:
            res.dual_thres_gt_labels = dual_thres_res.gt_labels
        res.lin_score_arrs[uncertainty], res.pred_prob_arrs[uncertainty] = (
            lin_scores,
            pred_probs,
        )
        res.pos_subscore_arrs[uncertainty], res.neg_subscore_arrs[uncertainty] = (
            pos_subscores,
            neg_subscores,
        )
        res.score_stats.loc[uncertainty, "mean_lin"] = np.mean(lin_scores)
        res.score_stats.loc[uncertainty, "std_lin"] = np.std(lin_scores)
        res.score_stats.loc[uncertainty, "mean_probs"] = np.mean(pred_probs)
        res.score_stats.loc[uncertainty, "std_probs"] = np.std(pred_probs)
    return res
