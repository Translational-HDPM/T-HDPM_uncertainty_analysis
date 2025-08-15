"""
Functions for post-processing (visualization and downstream analysis) of simulation results.
"""

from itertools import permutations
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import (
    inset_axes,
    mark_inset,
)
from sklearn.metrics import confusion_matrix, jaccard_score

from .dtypes import NumpyFloat32Array1D, NumpyFloat32Array2D


def get_differential_classification(
    gt_labels: pd.Series, pred_labels_dict: dict[int, pd.Series], labels: list[str]
) -> pd.DataFrame:
    """
    Calculate the differential classification percentages. Each row in the
    output DataFrame represents an uncertainty level and each column represents
    a true label. The values in the DataFrame indicate the percentage of
    instances with a specific true label that were misclassified as *any other*
    label at a given uncertainty level.

    Parameters
    ----------
    gt_labels
        A pandas Series containing the ground truth labels. These labels should
        be integers corresponding to the indices of the `labels` list.
    pred_labels_dict
        A dictionary where keys are uncertainty levels (integers) and values
        are pandas Series containing the predicted labels for each instance
        at that specific uncertainty level. These predicted labels should also
        be integers corresponding to the indices of the `labels` list.
    labels
        A list of strings representing the names of the classes. The order of
        these labels should correspond to the integer labels used in
        `gt_labels` and `pred_labels_dict`.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with uncertainty levels as the index and class names
        (from `labels`) as columns. Each cell `(u, l)` contains the percentage
        of instances with true label `l` that were misclassified at uncertainty
        level `u`. The values are scaled by 100 to represent percentages.

    Examples
    --------
    >>> gt = pd.Series([0, 0, 1, 1, 0, 1])
    >>> pred_dict = {
    ...     10: pd.Series([0, 1, 1, 0, 0, 1]),
    ...     20: pd.Series([1, 1, 0, 0, 1, 0])
    ... }
    >>> class_names = ['NCI', 'AD']
    >>> df = get_differential_classification(gt, pred_dict, class_names)
    >>> print(df)
           NCI        AD
    10   33.333333   0.000000
    20  100.000000  100.000000
    """
    diff_cls_df = pd.DataFrame(
        index=list(pred_labels_dict.keys()),
        columns=labels,
        data=np.zeros(shape=(len(list(pred_labels_dict.keys())), len(labels))),
    )
    diff_cls_df.loc[0, :] = np.zeros(len(labels))
    for uncert in pred_labels_dict:
        for true_label, fake_label in permutations(range(len(labels)), 2):
            subset = pred_labels_dict[uncert][gt_labels == true_label] == fake_label
            diff_cls_df.loc[uncert, labels[true_label]] += subset.sum()
    for i, label in enumerate(labels):
        diff_cls_df[label] = diff_cls_df[label] / (gt_labels == i).sum() * 100
    diff_cls_df = diff_cls_df.loc[np.sort(diff_cls_df.index), :]
    return diff_cls_df


def plot_confusion_matrix(
    cnf_mat: NumpyFloat32Array2D, categories: list[str], title: Optional[str] = None
) -> None:
    """
    Plot confusion matrix for simulation output.

    Parameters
    ----------
    cnf_mat
        Confusion matrix as a NumPy array
    categories
        String labels of categories in the order of appearance in the NumPy array
    title
        An optional title for the plot.

    Raises
    ------
    ValueError:
        Dimension of confusion matrix does not match the number of categories.
    """
    if cnf_mat.shape[0] != len(categories):
        raise ValueError(
            "Dimension of confusion matrix does not match the number of categories."
        )
    cnf_mat_df = pd.DataFrame(
        cnf_mat.astype(np.int64), index=categories, columns=categories
    )
    plt.figure(figsize=(8, 8))
    sns.heatmap(cnf_mat_df, annot=True, cbar=False, fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Confusion matrix")
    plt.show()


def display_differential_classification_results_one_threshold(
    *, ad_diff_cls: int, nci_diff_cls: int, gt_probs: NumpyFloat32Array1D, thres: float
) -> None:
    """
    Calculate metrics of differential classification and display results (single threshold).

    Parameters
    ----------
    ad_diff_cls
        Number of subjects differentially classified in the AD (Alzheimer's Disease) category
    nci_diff_cls
        Number of subjects differentially classified in the NCI (Non-Cognitively Impaired) category
    gt_probs
        Classifier probability scores for actual TPM values as a NumPy array
    thres
        Probability threshold for the binary classifier
    """
    num_nci = (thres > gt_probs).sum()
    num_ad = (thres <= gt_probs).sum()
    print(
        f"{ad_diff_cls / num_ad * 100:.2f} % simulated subjects were "
        "differentially classified from the Alzheimer's disease category."
    )
    print(
        f"{nci_diff_cls / num_nci * 100:.2f} % simulated subjects were "
        "differentially classified from the NCI category."
    )
    print(
        f"{(ad_diff_cls + nci_diff_cls) / len(gt_probs) * 100:.2f} % simulated"
        " subjects were differentially classified between AD and NCI categories."
    )
    print(
        "Total number of differentially classified individuals: "
        f"{(ad_diff_cls + nci_diff_cls)}"
    )


def display_differential_classification_results_two_thresholds(
    *,
    ad_diff_cls: int,
    int_diff_cls: int,
    nci_diff_cls: int,
    gt_probs: NumpyFloat32Array1D,
    thres_low: float,
    thres_high: float,
) -> None:
    """
    Calculate metrics of differential classification and display results (two
    thresholds).

    Parameters
    ----------
    ad_diff_cls
        Number of subjects differentially classified in the AD (Alzheimer's Disease) category
    int_diff_cls
        Number of subjects differentially classified in the intermediate category
    nci_diff_cls
        Number of subjects differentially classified in the NCI (Non-Cognitively Impaired) category
    gt_probs
        Classifier probability scores for actual TPM values as a NumPy array
    thres_low
        Probability threshold for the binary classifier between NCI and Intermediate
    thres_high
        Probability threshold for the binary classifier between Intermediate and AD
    """
    num_nci = (thres_low > gt_probs).sum()
    num_int = ((thres_low <= gt_probs) & (gt_probs < thres_high)).sum()
    num_ad = (thres_high <= gt_probs).sum()
    print(
        f"{ad_diff_cls / num_ad * 100:.2f} % simulated subjects were"
        " differentially classified from the Alzheimer's disease category."
    )
    print(
        f"{int_diff_cls / num_int * 100:.2f} % simulated subjects were "
        "differentially classified from the intermediate category."
    )
    print(
        f"{nci_diff_cls / num_nci * 100:.2f} % simulated subjects were "
        "differentially classified from the NCI category."
    )
    print(
        "Fraction of simulated subjects differentially classified: Approximately"
        f" {(ad_diff_cls + int_diff_cls + nci_diff_cls) / len(gt_probs) * 100:.2f}%"
    )
    print(
        "Total number of differentially classified individuals: "
        f"{(ad_diff_cls + int_diff_cls + nci_diff_cls)}"
    )


def calculate_sensitivity_specificity_and_predictive_values(
    gt_labels: pd.Series, pred_labels: pd.Series, label_idx: int
) -> tuple[float, float, float, float]:
    """
    Calculates sensitivity, specificity and predictive values for a category for
    a given label_idx from supplied ground truth and predicted labels.

    Parameters
    ----------
    gt_labels
        Ground truth integer labels as a Pandas series.
    pred_labels
        Predicted integer labels as a Pandas series. Must have the same index as
        `gt_labels`.
    label_idx
        Integer label for the class for which we are calculating our metrics.

    Returns
    -------
    tuple[float, float, float, float]
        Sensitivity, specificity, positive and negative predictive values.

    Raises
    ------
    ValueError
        If the `label_idx` supplied is not in either of `gt_labels` or `pred_labels`.
    """
    if label_idx not in np.unique(gt_labels) or label_idx not in np.unique(pred_labels):
        raise ValueError("Label not found in one of gt_labels or pred_labels array")
    cnf_mat = confusion_matrix(gt_labels, pred_labels)
    tp = cnf_mat[label_idx, label_idx]
    tn = (
        np.sum(cnf_mat[:label_idx, :label_idx])
        + np.sum(cnf_mat[label_idx + 1 :, label_idx + 1 :])
        + np.sum(cnf_mat[label_idx + 1 :, :label_idx])
        + np.sum(cnf_mat[:label_idx, label_idx + 1 :])
    )
    fp = np.sum(cnf_mat[:label_idx, label_idx]) + np.sum(
        cnf_mat[label_idx + 1 :, label_idx]
    )
    fn = np.sum(cnf_mat[label_idx, :label_idx]) + np.sum(
        cnf_mat[label_idx, label_idx + 1 :]
    )
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return sensitivity, specificity, ppv, npv


def calculate_subject_wise_agreement(
    *,
    gt_series_dict: dict[int, pd.Series],
    pred_series_dict: dict[int, pd.Series],
    uncertainties: list[int],
    num_patients: int = 243,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """
    Calculate the percent of simulated predictions that agree with the actual
    prediction for each subject.

    Parameters
    ----------
    gt_series_dict
        Dictionary containing labels for actual data for subjects predicted
        by the classifier. The keys are percent uncertainties and the corresponding
        values are Pandas series with the labels (ordinal encoding, i.e. 0 for NCI,
        1 for AD, etc.)
    pred_series_dict
        Dictionary containing labels for simulated data for subjects predicted by
        the classifier. The keys are percent uncertainties and the corresponding
        values are Pandas series with the labels (ordinal encoding, i.e. 0 for NCI,
        1 for AD, etc.)
    uncertainties
        List of integer values representing percent uncertainty values simulated.
    num_patients
        Number of subjects
    n_samples
        Number of simulated points per subject

    Returns
    -------
    pd.DataFrame
        A Pandas Dataframe with percent values indicating what percent of predictions
        for simulated points agree with the actual classification.
    """
    subj_wise_agreement = pd.DataFrame(
        index=gt_series_dict[uncertainties[0]].index,
        columns=[f"{uncert}% uncertainty" for uncert in uncertainties],
    )
    for uncertainty in uncertainties:
        gt, preds = gt_series_dict[uncertainty], pred_series_dict[uncertainty]
        preds = preds[gt.index]
        subj_wise_agreement.loc[:, f"{uncertainty}% uncertainty"] = (
            (np.array(gt.values.tolist()) == np.array(preds.values.tolist())).sum(
                axis=1
            )
            / n_samples
            * 100
        )
    subj_wise_agreement.index.name = "Patient ID"
    return subj_wise_agreement


def calculate_subject_wise_disagreement(
    *,
    gt_series_dict: dict[int, pd.Series],
    pred_series_dict: dict[int, pd.Series],
    uncertainties: list[int],
    categories: list[str],
    num_patients: int = 243,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """
    Calculate the percent of simulated predictions that do not agree with the actual
    prediction for each subject.

    Parameters
    ----------
    gt_series_dict
        Dictionary containing labels for actual data for subjects predicted
        by the classifier. The keys are percent uncertainties and the corresponding
        values are Pandas series with the labels (ordinal encoding, i.e. 0 for NCI,
        1 for AD, etc.)
    pred_series_dict
        Dictionary containing labels for simulated data for subjects predicted by
        the classifier. The keys are percent uncertainties and the corresponding
        values are Pandas series with the labels (ordinal encoding, i.e. 0 for NCI,
        1 for AD, etc.)
    uncertainties
        List of integer values representing percent uncertainty values simulated.
    categories
        List of strings representing categories for the classifier.
    num_patients
        Number of subjects
    n_samples
        Number of simulated points per subject

    Returns
    -------
    pd.DataFrame
        A Pandas Dataframe with category-wise percent values indicating what percent
        of simulated points got misclassified as that category.
    """
    subj_wise_disagreement = pd.DataFrame(
        index=gt_series_dict[uncertainties[0]].index,
        columns=[
            f"{uncert}% uncertainty: % misclassified as {category}"
            for uncert in uncertainties
            for category in categories
        ],
    )
    for uncertainty in uncertainties:
        gt, preds = gt_series_dict[uncertainty], pred_series_dict[uncertainty]
        preds = preds[gt.index]
        for i, cat in enumerate(categories):
            subj_wise_disagreement.loc[
                :, f"{uncertainty}% uncertainty: % misclassified as {cat}"
            ] = np.round(
                (np.array(preds.values.tolist()) == i).sum(axis=1) / n_samples * 100, 2
            )
        for patient_id in subj_wise_disagreement.index:
            subj_wise_disagreement.loc[
                patient_id,
                f"{uncertainty}% uncertainty: % misclassified as {categories[gt[patient_id][0]]}",
            ] = np.nan
    return subj_wise_disagreement


def build_sensitivity_specificity_df(
    pathos_df: pd.DataFrame,
    gt_probs_ser: pd.Series,
    label: Literal["AD", "NCI"],
) -> pd.DataFrame:
    """
    Builds a DataFrame of sensitivity and specificity through a range of
    thresholds (1 to 99) and calculates the sensitivity and specificity for a
    specified label at each threshold, returning these results in a pandas
    DataFrame.

    Parameters
    ----------
    pathos_df : pd.DataFrame
        A pandas DataFrame expected to have an 'index' that aligns with
        `gt_probs_ser` and a 'Disease' column with values like "AD" or "NCI".
    gt_probs_ser : pd.Series
        A pandas Series containing ground truth probabilities, typically ranging
        from 0 to 1.
    label : Literal["AD", "NCI"]
        The specific label ("AD" or "NCI") for which to calculate sensitivity
        and specificity across thresholds.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with columns "threshold", "sensitivity", and
        "specificity". Each row corresponds to a threshold from 1 to 99,
        and the respective sensitivity and specificity values for the given label.

    Raises
    ------
    ValueError
        If label is not 'AD' or 'NCI'.
    """
    if label not in ["AD", "NCI"]:
        raise ValueError("Invalid label.")
    sens_spec_df = pd.DataFrame(
        columns=["threshold", "sensitivity", "specificity", "ppv", "npv"]
    )
    sens_spec_df["threshold"] = np.arange(1, 100)
    temp = sens_spec_df["threshold"].apply(
        lambda thres: calculate_sensitivity_specificity_and_predictive_values(
            pathos_df["Disease"].apply(lambda x: 1 if x == "AD" else 0),
            gt_probs_ser[pathos_df.index].apply(
                lambda x: 1 if x >= thres / 100.0 else 0
            ),
            label_idx=1 if label == "AD" else 0,
        )
    )
    sens_spec_df.loc[:, "sensitivity"] = temp.apply(lambda x: x[0])
    sens_spec_df.loc[:, "specificity"] = temp.apply(lambda x: x[1])
    sens_spec_df.loc[:, "ppv"] = temp.apply(lambda x: x[2])
    sens_spec_df.loc[:, "npv"] = temp.apply(lambda x: x[3])
    return sens_spec_df


def get_threshold(
    value: float,
    sens_spec_df: pd.DataFrame,
    metric: Literal["sensitivity", "specificity"] = "sensitivity",
) -> float:
    """
    Retrieves the probability threshold corresponding to a target
    sensitivity/specificity.

    This function finds the highest threshold from a sensitivity/specificity
    DataFrame that yields a sensitivity greater than or equal to the
    specified target sensitivity.

    Parameters
    ----------
    value
        The target sensitivity/specificity value (as a percentage, e.g., 90 for 90%)
        for which to find the corresponding threshold.
    sens_spec_df
        A pandas DataFrame, typically generated by `build_sensitivity_specificity_df`,
        containing "threshold" and "sensitivity" columns.
    metric
        Whether to use sensitivity or specificity. Defaults to sensitivity.

    Returns
    -------
    float
        The probability threshold (as an integer percentage, e.g., 50) that
        corresponds to the highest threshold where the metric is
        greater than or equal to the `value` input.

    Raises
    ------
    ValueError
        1. If the `value` provided is outside the range of values of the metric
        available in the `sens_spec_df`.
        2. If the `metric` specified is not "sensitivity" or "specificity".

    See Also
    --------
    build_sensitivity_specificity_df : Generates the DataFrame used by this function.
    """
    if metric not in ["sensitivity", "specificity"]:
        raise ValueError("Metric must be either sensitivity or specificity.")
    if not sens_spec_df[metric].min() < value / 100 < sens_spec_df[metric].max():
        raise ValueError(
            f"{metric} out of bounds. Choose a {metric} between"
            + f" {sens_spec_df[metric].min() * 100:.2f}% and"
            + f" {sens_spec_df[metric].max() * 100:.2f}%"
        )
    # If sensitivity is a decreasing function of threshold, return last threshold value
    if (
        sens_spec_df.loc[0, metric]
        > sens_spec_df.loc[sens_spec_df.shape[0] - 1, metric]
    ):
        return sens_spec_df.loc[sens_spec_df[metric] >= value / 100, "threshold"].iloc[
            -1
        ]
    # Else return first threshold value
    return sens_spec_df.loc[sens_spec_df[metric] >= value / 100, "threshold"].iloc[0]


def plot_bland_altman(
    arr_1: NumpyFloat32Array1D,
    arr_2: NumpyFloat32Array1D,
    title: str,
    *,
    save: bool = False,
    show: bool = True,
) -> Axes:
    """
    Generate a Bland-Altman plot for two sets of measurements `arr_1` and `arr_2`.

    Parameters
    ----------
    arr_1
        An np.ndarray of float32 values representing a set of measurements from
        an assay.
    arr_2
        Another np.ndarray of float32 values representing a set of measurements
        from a second assay. arr_1 and arr_2 should be of the same shape.
    title
        Title of the plot
    save
        Whether to save the generated plot. If True, saves the plot as a PNG image
        of the same name as the title.
    show
        Whether to display the generated plot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib `Axes` object corresponding to the generated plot.

    Raises
    ------
    ValueError
        If the shapes of arr_1 and arr_2 mismatch, a `ValueError` is raised.
    """
    if arr_1.shape != arr_2.shape:
        raise ValueError("Shape mismatch between arr_1 and arr_2.")

    # Compute the average and difference of the two methods
    mean_measurements = (arr_1 + arr_2) / 2.0
    differences = arr_1 - arr_2

    # Compute statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # Limits of agreement (mean difference ± 1.96*SD)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Plot Bland-Altman plot
    plt.scatter(mean_measurements, differences, color="blue", alpha=0.7)
    plt.axhline(
        mean_diff, color="gray", linestyle="--", label=f"Mean diff = {mean_diff:.2f}"
    )
    plt.axhline(
        loa_upper, color="red", linestyle="--", label=f"Upper LoA = {loa_upper:.2f}"
    )
    plt.axhline(
        loa_lower, color="red", linestyle="--", label=f"Lower LoA = {loa_lower:.2f}"
    )

    plt.xlabel("Mean of two measurements")
    plt.ylabel("Difference between measurements")
    plt.ylim([1.5 * loa_lower, 1.5 * loa_upper])
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(f"{title}.png")
    if show:
        plt.show()


def plot_v_plot(
    subj_wise_agreement: pd.DataFrame,
    gt_probs: pd.Series,
    uncertainties: Sequence[int],
    title: str,
    show_axis_labels: bool = True,
    show_legend: bool = False,
) -> Axes:
    """
    Creates a v-plot between the agreement of simulated scores and classifier scores for
    subjects against the inferent probability scores of the subjects.

    Parameters
    ----------
    subj_wise_agreement
        A dataframe containing percent agreement between simulated and classifier scores
        at different percentages of simulated uncertainties.
    gt_probs
        Probability values from the classifier for the original data of TPM for the patients.
    uncertainties
        Simulated percentage values of uncertainties.
    title
        Title for the generated plot.
    show_axis_labels
        Whether to show axis labels in the generated plot.
    show_legend
        Whether to show a legend in the generated plot.

    Returns
    -------
    matplotlib.axes.Axes
        A matplotlib `Axes` object corresponding to the generated plot.
    """
    gt_probs = gt_probs.sort_values()
    _temp = subj_wise_agreement.loc[gt_probs.index, :]
    _max_alpha = [
        np.min(uncertainties),
        np.median(uncertainties),
        np.max(uncertainties),
    ]
    for uncert in uncertainties:
        if uncert in _max_alpha:
            plt.plot(
                gt_probs,
                _temp[f"{uncert}% uncertainty"],
                label=f"{uncert}% uncertainty",
            )
        else:
            plt.plot(
                gt_probs,
                _temp[f"{uncert}% uncertainty"],
                label=f"{uncert}% uncertainty",
                alpha=0.2,
            )
    plt.title(title)
    if show_axis_labels:
        plt.xlabel("Probability score")
        plt.ylabel(
            "Percent agreement between simulated and\n inferent scores for subjects"
        )
    if show_legend:
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(uncertainties) // 3
        )


def generate_waterfall_plot(
    *,
    threshold: float,
    probs: pd.Series,
    color_labels_data: pd.Series,
    labels: dict[int, str],
    colors: list[str],
    title: str,
    legend_title: str = "",
    save: bool = False,
) -> Figure:
    """
    Creates a waterfall plot showing a comparison between predictions by a binary
    classifier against the "true classes" specified by the `color_labels_data`.
    In `color_labels_data` the classes are integer values for which the `labels`
    dictionary provides the string representations.

    Parameters
    ----------
    threshold
        The probability cut-point which acts as the binary decision point.
    probs
        Probability values from the classifier for the original data of TPM for the patients.
    color_labels_data
        A Pandas series with integer labels corresponding to classification according to some
        criterion, e.g. modeled measurement uncertainty.
    labels
        Dictionary containing string labels corresponding to integer values for classes in
        `color_labels_data`.
    colors
        Hex codes for colors for bars for each unique label.
    title
        Title for the plot.
    legend_title
        Title for the legend.
    save
        Whether to save the generated plot. If specified as true, saves the plot as a PNG image
        of the same name as the title.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure corresponding to the generated plot.

    Raises
    ------
    ValueError
        1. If `threshold` is not between 0 and 1.
        2. If the indexes of the `probs` and `color_labels_data` Series do not match. This is
           required to ensure matching probabilities with the color labels when they are
           combined into a dataframe.
        3. If the number of colors specified is not the same as the number of unique labels.
    """

    # Check for threshold to be between 0 and 1
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    # Check for the indexes of probs and color_labels to be the same
    if not probs.index.tolist() == color_labels_data.index.tolist():
        raise ValueError("Indexes of probs and color_labels_data must be identical.")
    if not len(colors) == len(labels):
        raise ValueError(
            "Must supply a list of colors of same length as the number of unique labels."
        )

    probs_df = pd.DataFrame(index=probs.index)
    probs_df["probs"] = probs
    probs_df["color_labels"] = color_labels_data
    probs_df.sort_values(by="probs", inplace=True)

    probs_df["x"] = np.linspace(-1, 40, probs.shape[0])
    probs_df["probs"] -= threshold

    fig = plt.figure(figsize=(12, 8))
    unique_labels = probs_df["color_labels"].unique()

    for label, color in zip(unique_labels, colors):
        filt = probs_df["color_labels"] == label
        plt.bar(
            probs_df.loc[filt, "x"],
            probs_df.loc[filt, "probs"],
            width=0.2,
            color=color,
            label=labels[label],
        )

    curr_yticks = plt.gca().get_yticks()
    plt.xticks([])
    plt.yticks(curr_yticks, np.round(curr_yticks + threshold, 2))
    plt.ylabel("Classifier score")
    plt.legend(title=legend_title)
    plt.title(title, fontsize=15)
    if save:
        plt.savefig(f"{title}.png")
    plt.show()
    return fig


def calculate_jaccard_index(
    *, labels: list[str], gt_labels: pd.Series, pred_labels_dict: dict[int, pd.Series]
) -> pd.DataFrame:
    """
    Calculate the Jaccard similarity score for a set of predictions
    against ground truth labels. The predictions are provided in a dictionary,
    where each entry corresponds to a different level of uncertainty. The
    function returns a DataFrame summarizing the Jaccard index for each class
    label across all uncertainty levels.

    Parameters
    ----------
    labels
        A list of strings representing the class labels to be evaluated.
    gt_labels
        A pandas Series containing the true ground truth labels.
    pred_labels_dict
        A dictionary mapping an uncertainty level (integer key) to a pandas
        Series of predicted labels. The keys represent the uncertainty
        threshold, and the values are the corresponding predictions.

    Returns
    -------
    pd.DataFrame
        A DataFrame where rows are indexed by uncertainty levels and columns
        are indexed by class labels. Each cell `(i, j)` contains the
        Jaccard index for class `j` at uncertainty level `i`. The row for
        uncertainty `0` is initialized to all ones as a baseline.
    """
    jaccard_index = pd.DataFrame(
        index=list(pred_labels_dict.keys()), columns=list(labels)
    )
    jaccard_index.loc[0, :] = np.ones(len(labels))
    for uncert in pred_labels_dict:
        jaccard_index.loc[uncert, :] = jaccard_score(
            pred_labels_dict[uncert].values,
            gt_labels.values,
            average=None,
        )
    return jaccard_index


def plot_jaccard_index_plot(
    *,
    labels_dict_single_thres: dict[str, str],
    labels_dict_dual_thres: dict[str, str],
    gt_labels_single_thres: pd.Series,
    gt_labels_dual_thres: pd.Series,
    pred_labels_dict_single_thres: dict[int, pd.Series],
    pred_labels_dict_dual_thres: dict[int, pd.Series],
    single_thres_plot_title: str,
    dual_thres_plot_title: str,
    figure_title: str,
    save: bool = False,
) -> Figure:
    """
    Generates a figure with two subplots, each showing the Jaccard index for
    different class labels as a function of an uncertainty percentage. It is
    designed to compare the performance of a single-threshold classification
    method against a dual-threshold method.

    Parameters
    ----------
    labels_dict_single_thres
        Dictionary mapping class labels to plot colors for the single-threshold
        (left) plot. e.g., `{'AD': 'b', 'NCI': 'r'}`.
    labels_dict_dual_thres
        Dictionary mapping class labels to plot colors for the dual-threshold
        (right) plot. Must have consistent colors with `labels_dict_single_thres`.
    gt_labels_single_thres
        A pandas Series containing the ground truth labels for the
        single-threshold scenario.
    gt_labels_dual_thres
        A pandas Series containing the ground truth labels for the
        dual-threshold scenario.
    pred_labels_dict_single_thres
        Dictionary mapping uncertainty levels (int) to predicted labels
        (pd.Series) for the single-threshold scenario.
    pred_labels_dict_dual_thres
        Dictionary mapping uncertainty levels (int) to predicted labels
        (pd.Series) for the dual-threshold scenario.
    single_thres_plot_title
        The title for the left subplot (single-threshold).
    dual_thres_plot_title
        The title for the right subplot (dual-threshold).
    figure_title
        The main title for the entire figure.
    save
        If True, the figure is saved to a PNG file named after the
        `figure_title`. Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure corresponding to the generated plot.

    Raises
    ------
    ValueError
        If a class label has a different color mapping between
        `labels_dict_single_thres` and `labels_dict_dual_thres`.

    See Also
    --------
    calculate_jaccard_index : The function used to compute the Jaccard scores.
    """
    for label in labels_dict_single_thres:
        if labels_dict_single_thres[label] != labels_dict_dual_thres[label]:
            raise ValueError(
                f"Difference in linestyle between single and dual threshold plots for label '{label}'"
            )
    fig, axs = plt.subplots(figsize=(16, 7), nrows=1, ncols=2, sharex=True, sharey=True)
    plt.subplot(121)
    jac_idx_df = calculate_jaccard_index(
        labels=list(labels_dict_single_thres.keys()),
        gt_labels=gt_labels_single_thres,
        pred_labels_dict=pred_labels_dict_single_thres,
    )
    x_vals = np.sort(jac_idx_df.index.values)
    for col in jac_idx_df.columns:
        plt.plot(
            x_vals,
            jac_idx_df.loc[x_vals, col],
            label=col,
            color=labels_dict_single_thres[col],
        )

    plt.title(single_thres_plot_title)

    plt.subplot(122)
    jac_idx_df = calculate_jaccard_index(
        labels=list(labels_dict_dual_thres.keys()),
        gt_labels=gt_labels_dual_thres,
        pred_labels_dict=pred_labels_dict_dual_thres,
    )
    x_vals = np.sort(jac_idx_df.index.values)
    for col in jac_idx_df.columns:
        plt.plot(
            x_vals,
            jac_idx_df.loc[x_vals, col],
            label=col,
            color=labels_dict_dual_thres[col],
        )
    plt.title(dual_thres_plot_title)
    fig.text(
        0.09,
        0.5,
        "Jaccard index",
        va="center",
        ha="center",
        rotation="vertical",
    )
    fig.text(0.5, 0.05, "Pct. uncertainty", va="center", ha="center")
    leg_handles, leg_labels = plt.gca().get_legend_handles_labels()
    fig.legend(
        leg_handles,
        leg_labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.suptitle(figure_title, fontsize=14)
    if save:
        fig.savefig(f"{figure_title}.png")
    return fig


def plot_differential_classification_results(
    *,
    labels_dict_single_thres: dict[str, str],
    labels_dict_dual_thres: dict[str, str],
    gt_labels_single_thres: pd.Series,
    gt_labels_dual_thres: pd.Series,
    pred_labels_dict_single_thres: dict[int, pd.Series],
    pred_labels_dict_dual_thres: dict[int, pd.Series],
    single_thres_plot_title: str,
    dual_thres_plot_title: str,
    figure_title: str,
    save: bool = False,
) -> Figure:
    """
    Plots differential classification results for single and dual threshold scenarios.

    Generates a figure with two subplots, each showing the percentage of
    differentially classified subjects for various uncertainty levels, based
    on single and dual threshold scenarios.

    Parameters
    ----------
    labels_dict_single_thres
        Dictionary mapping class labels to plot colors for the single-threshold
        (left) plot. e.g., `{'AD': 'b', 'NCI': 'r'}`.
    labels_dict_dual_thres
        Dictionary mapping class labels to plot colors for the dual-threshold
        (right) plot. Must have consistent colors with `labels_dict_single_thres`.
    gt_labels_single_thres
        A pandas Series containing the ground truth labels for the
        single-threshold scenario.
    gt_labels_dual_thres
        A pandas Series containing the ground truth labels for the
        dual-threshold scenario.
    pred_labels_dict_single_thres
        Dictionary mapping uncertainty levels (int) to predicted labels
        (pd.Series) for the single-threshold scenario.
    pred_labels_dict_dual_thres
        Dictionary mapping uncertainty levels (int) to predicted labels
        (pd.Series) for the dual-threshold scenario.
    single_thres_plot_title
        The title for the left subplot (single-threshold).
    dual_thres_plot_title
        The title for the right subplot (dual-threshold).
    figure_title
        The main title for the entire figure.
    save
        If True, the figure is saved to a PNG file named after the
        `figure_title`. Default is False.

    Raises
    ------
    ValueError
        If a class label has a different color mapping between
        `labels_dict_single_thres` and `labels_dict_dual_thres`.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure generated.

    See Also
    --------
    get_differential_classification : Calculates the underlying data for the plots.
    """
    for label in labels_dict_single_thres:
        if labels_dict_single_thres[label] != labels_dict_dual_thres[label]:
            raise ValueError(
                f"Difference in linestyle between single and dual threshold plots for label '{label}'"
            )
    fig, axs = plt.subplots(figsize=(16, 7), nrows=1, ncols=2, sharex=True, sharey=True)

    scenarios = ["single threshold", "dual threshold"]
    label_counts_dict = dict.fromkeys(scenarios)
    for i, (labels_dict, gt_labels, pred_labels_dict, plot_title) in enumerate(
        [
            (
                labels_dict_single_thres,
                gt_labels_single_thres,
                pred_labels_dict_single_thres,
                single_thres_plot_title,
            ),
            (
                labels_dict_dual_thres,
                gt_labels_dual_thres,
                pred_labels_dict_dual_thres,
                dual_thres_plot_title,
            ),
        ]
    ):
        plt.subplot(1, 2, i + 1)
        label_counts_dict[scenarios[i]] = {
            label: (gt_labels == i).sum() for i, label in enumerate(labels_dict.keys())
        }
        results = get_differential_classification(
            gt_labels,
            pred_labels_dict,
            list(labels_dict.keys()),
        )
        for cat in results.columns:
            plt.plot(
                results.index, results.loc[:, cat], label=cat, color=labels_dict[cat]
            )
        plt.title(plot_title)

    leg_handles, leg_labels = plt.gca().get_legend_handles_labels()
    fig.legend(
        leg_handles,
        leg_labels,
        loc="upper center",
        ncol=len(leg_labels),
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.text(0.5, 0.05, "Pct. uncertainty", va="center", ha="center")
    fig.text(
        0.09,
        0.5,
        "Percent of subjects within category differentially classified",
        va="center",
        ha="center",
        rotation="vertical",
    )
    fig.suptitle(figure_title, fontsize=14)
    for i, scenario in enumerate(scenarios):
        label_counts = label_counts_dict[scenario]
        fig.text(
            0.5,
            -0.06 * (i + 1),
            f"Classifier predictions ({scenario}): "
            + ", ".join([f"{label_counts[label]} {label}" for label in label_counts])
            + " subjects",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                alpha=0.8,
            ),
        )
    return fig


def plot_histogram_of_simulated_and_real_subject_probability_scores(
    gt_probs: NumpyFloat32Array1D,
    pred_probs: NumpyFloat32Array1D,
    uncertainty: int,
    plot_inset: bool = True,
) -> Figure:
    """Plot histograms of simulated and real subject probability scores.

    This function generates a figure with two histograms: one for classifier
    (`gt_probs`) probability scores (from unsimulated subject data) and
    one for probability scores of the classifier for simulated subject data.
    It also includes a zoomed-in inset plot for a detailed view of a specific
    probability range.

    Parameters
    ----------
    gt_probs
        A 1D NumPy array containing the classifier probability
        scores for real subjects.
    pred_probs
        A 1D NumPy array containing the classifier probability
        scores for simulated subjects.
    uncertainty
        The percentage uncertainty value associated with the simulated data.
    plot_inset
        Whether to plot an inset of the region between probabilities 0.1 and 0.9.
        Defaults to True.

    Returns
    -------
    matplotlib.figure.Figure
        A Matplotlib Figure object containing the generated histograms and inset.
    """
    nbins = 30

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(
        pred_probs,
        color="b",
        alpha=0.3,
        fill=True,
        bins=nbins,
        stat="density",
        label="Simulated subjects",
        ax=ax,
    )
    sns.histplot(
        gt_probs,
        color="r",
        alpha=0.3,
        fill=True,
        bins=nbins,
        stat="density",
        label="Real subjects",
        ax=ax,
    )
    plt.xlabel("Classifier score (probability)")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.title(
        f"Histogram of probability scores from \nsimulated and real subjects at {uncertainty}% simulated uncertainty."
    )
    if plot_inset:
        # Make zoomed inset
        inset_axs = inset_axes(ax, loc="center", width="50%", height="50%")
        sns.histplot(
            pred_probs,
            color="b",
            alpha=0.3,
            fill=True,
            bins=nbins,
            stat="density",
            label="Simulated subjects",
            ax=inset_axs,
        )
        sns.histplot(
            gt_probs,
            color="r",
            alpha=0.3,
            fill=True,
            bins=nbins,
            stat="density",
            label="Real subjects",
            ax=inset_axs,
        )
        x_min, x_max = 0.1, 0.9
        hist_data, bins = np.histogram(
            pred_probs,
            bins=nbins,
            density=True,
        )
        y_min, y_max = (
            0,
            1.5 * hist_data[(bins[:-1] >= x_min) & (bins[:-1] <= x_max)].max(),
        )
        inset_axs.set_xlim(x_min, x_max)
        inset_axs.set_ylim(y_min, y_max)

        plt.xlabel("")
        plt.ylabel("")

        mark_inset(
            ax,
            inset_axs,
            loc1=1,
            loc2=2,
            fc="none",
            ec="gray",
        )

    return fig
