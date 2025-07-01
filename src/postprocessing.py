"""
Functions for post-processing (visualization and downstream analysis) of simulation results.
"""

from typing import Optional, Sequence
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import jaccard_score

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
    gt_labels : pd.Series
        A pandas Series containing the ground truth labels. These labels should
        be integers corresponding to the indices of the `labels` list.
    pred_labels_dict : dict[int, pd.Series]
        A dictionary where keys are uncertainty levels (integers) and values
        are pandas Series containing the predicted labels for each instance
        at that specific uncertainty level. These predicted labels should also
        be integers corresponding to the indices of the `labels` list.
    labels : list[str]
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
    for uncert in pred_labels_dict:
        for true_label, fake_label in permutations(range(len(labels)), 2):
            subset = pred_labels_dict[uncert][gt_labels == true_label] == fake_label
            diff_cls_df.loc[uncert, labels[true_label]] += subset.sum()
    for i, label in enumerate(labels):
        diff_cls_df[label] = diff_cls_df[label] / (gt_labels == i).sum() * 100
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


def calculate_sens_spec_dual_threshold(cnf_mat: NumpyFloat32Array2D) -> str:
    """
    Calculates and displays sensitivity and specificity for Alzheimer's disease (AD)
    and NCI categories from a confusion matrix for results from dual threshold simulations.

    Parameters
    ----------
    cnf_mat
        Confusion matrix as a NumPy array

    Returns
    -------
    str
        Markdown table representing results of the sensitivity and specificity calculations

    Raises
    ------
    ValueError:
        Confusion matrix should be 3x3.
    """
    if cnf_mat.shape[0] != 3:
        raise ValueError("Confusion matrix should be 3x3.")
    # AD
    ad_tp = cnf_mat[2, 2]
    ad_tn = np.sum(cnf_mat[:2, :2])
    ad_fp = np.sum(cnf_mat[:2, 2])
    ad_fn = np.sum(cnf_mat[2, :2])
    sensitivity_ad = ad_tp / (ad_tp + ad_fn)
    specificity_ad = ad_tn / (ad_tn + ad_fp)
    ppv_ad = ad_tp / (ad_tp + ad_fp)
    npv_ad = ad_tn / (ad_tn + ad_fn)

    # NCI
    nci_tp = cnf_mat[0, 0]
    nci_tn = np.sum(cnf_mat[1:, 1:])
    nci_fp = np.sum(cnf_mat[1:, 0])
    nci_fn = np.sum(cnf_mat[0, 1:])
    sensitivity_nci = nci_tp / (nci_tp + nci_fn)
    specificity_nci = nci_tn / (nci_tn + nci_fp)
    ppv_nci = nci_tp / (nci_tp + nci_fp)
    npv_nci = nci_tn / (nci_tn + nci_fn)

    string = f"""
    | **Metric**    | **AD (%)** | **NCI (%)** |
    |---------------|------------|-------------|
    | Sensitivity   | {sensitivity_ad * 100:.2f} | {sensitivity_nci * 100:.2f}|
    | Specificity   | {specificity_ad * 100:.2f} | {specificity_nci * 100:.2f} |
    | PPV           | {ppv_ad * 100:.2f} | {ppv_nci * 100:.2f} |
    | NPV           | {npv_ad * 100:.2f} | {npv_nci * 100:.2f} |
    """
    return string


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


def plot_bland_altman(
    arr_1: NumpyFloat32Array1D,
    arr_2: NumpyFloat32Array1D,
    title: str,
    *,
    save: bool = False,
    show: bool = True,
) -> None:
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
    None

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
    plt.figure(figsize=(8, 5))
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
        return
    plt.close()


def plot_v_plot(
    subj_wise_agreement: pd.DataFrame,
    gt_probs: pd.Series,
    uncertainties: Sequence[int],
    title: str,
    show_axis_labels: bool = True,
    show_legend: bool = False,
) -> None:
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
    None
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
) -> None:
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
    None

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

    plt.figure(figsize=(12, 8))
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
) -> None:
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
    None
        This function does not return any value. It displays a matplotlib plot.

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
    fig, axs = plt.subplots(figsize=(16, 6), nrows=1, ncols=2, sharex=True, sharey=True)
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

    plt.legend()
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
    plt.legend()
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
    fig.suptitle(figure_title)
    plt.show()
    if save:
        fig.savefig(f"{figure_title}.png")


def plot_differential_classification_results(
    *,
    gt_labels: pd.Series,
    one_sim_mismatch_pred_labels_dict: dict[int, pd.Series],
    ten_pct_sim_mismatch_pred_labels_dict: dict[int, pd.Series],
    labels_dict: dict[list[str], str],
    figure_title: str,
) -> None:
    """
    Plots differential classification results for single and dual threshold scenarios.

    This function generates a two-subplot figure displaying the percentage of
    differentially classified subjects for various uncertainty levels, based
    on single and dual threshold scenarios.

    Parameters
    ----------
    gt_labels
        A pandas Series containing the ground truth labels. These labels should
        be integers corresponding to the indices of the `labels` list.
    one_sim_mismatch_pred_labels_dict
        A dictionary where keys are uncertainty levels (integers) and values
        are pandas Series containing predicted labels for the "at least 1
        simulation mismatch" scenario.
    ten_pct_sim_mismatch_pred_labels_dict
        A dictionary where keys are uncertainty levels (integers) and values
        are pandas Series containing predicted labels for the "at least 10%
        of simulations mismatch" scenario.
    labels_dict
        A dictionary where the keys represent the names of the classes and the
        corresponding values are the line styles in the lineplot. The order of
        these labels (in the keys) should correspond to the integer labels used in the
        prediction dictionaries.
    figure_title
        The main title for the entire figure.

    Returns
    -------
    None
        This function does not return any value. It displays a matplotlib figure.

    See Also
    --------
    get_differential_classification : Calculates the underlying data for the plots.
    """
    label_counts = {
        label: (gt_labels == i).sum() for i, label in enumerate(labels_dict.keys())
    }
    fig = plt.figure(figsize=(16, 6))
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(16, 6))
    plt.subplot(121)

    results = get_differential_classification(
        gt_labels,
        one_sim_mismatch_pred_labels_dict,
        list(labels_dict.keys()),
    )
    for cat in results.columns:
        plt.plot(results.index, results.loc[:, cat], label=cat, color=labels_dict[cat])
    plt.title("At least 1 simulation mismatch")

    plt.subplot(122)
    results = get_differential_classification(
        gt_labels,
        ten_pct_sim_mismatch_pred_labels_dict,
        list(labels_dict.keys()),
    )
    for cat in results.columns:
        plt.plot(results.index, results.loc[:, cat], label=cat, color=labels_dict[cat])
    plt.title(
        "At least 10% of simulations mismatch",
    )

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
    fig.suptitle(figure_title)
    fig.text(
        0.5,
        -0.06,
        "Classifier predictions: "
        + ", ".join([f"{label_counts[label]} {label}" for label in label_counts])
        + " subjects",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.8
        ),
    )
