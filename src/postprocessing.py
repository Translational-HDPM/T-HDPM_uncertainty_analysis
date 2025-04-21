"""
Functions for post-processing (visualization and downstream analysis) of simulation results.
"""
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(cnf_mat: np.ndarray, categories: list[str], title: Optional[str] = None) -> None:
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
        raise ValueError("Dimension of confusion matrix does not match the number of categories.")
    cnf_mat_df = pd.DataFrame(cnf_mat.astype(np.int64), index=categories, columns=categories)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cnf_mat_df, annot=True, cbar=False, fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Confusion matrix")
    plt.show()

def display_differential_classification_results_one_threshold(*,
              ad_diff_cls: int, nci_diff_cls: int, 
              gt_probs: np.ndarray, thres: float) -> None:
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
    print(f"{ad_diff_cls / num_ad * 100:.2f} % simulated subjects were "
        "differentially classified from the Alzheimer's disease category.")
    print(f"{nci_diff_cls / num_nci * 100:.2f} % simulated subjects were "
        "differentially classified from the NCI category.")
    print(f"{(ad_diff_cls + nci_diff_cls) / len(gt_probs) * 100:.2f} % simulated"
        " subjects were differentially classified between AD and NCI categories.")
    print("Total number of differentially classified individuals: "
        f"{(ad_diff_cls + nci_diff_cls)}")

def display_differential_classification_results_two_thresholds(*,
       ad_diff_cls: int, 
       int_diff_cls: int, 
       nci_diff_cls: int,
       gt_probs: np.ndarray,
       thres_low: float,
       thres_high: float) -> None:
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
    print(f"{ad_diff_cls / num_ad * 100:.2f} % simulated subjects were"
        " differentially classified from the Alzheimer's disease category.")
    print(f"{int_diff_cls / num_int * 100:.2f} % simulated subjects were "
        "differentially classified from the intermediate category.")
    print(f"{nci_diff_cls / num_nci * 100:.2f} % simulated subjects were "
        "differentially classified from the NCI category.")
    print("Fraction of simulated subjects differentially classified: Approximately"
        f" {(ad_diff_cls + int_diff_cls + nci_diff_cls) / len(gt_probs) * 100:.2f}%")
    print("Total number of differentially classified individuals: "
        f"{(ad_diff_cls + int_diff_cls + nci_diff_cls)}")


def calculate_sens_spec_dual_threshold(cnf_mat: np.ndarray) -> str:
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

def calculate_subject_wise_agreement(*,
                                     gt_series_dict: dict[int, pd.Series],
                                     pred_series_dict: dict[int, pd.Series],
                                     uncertainties: list[int],
                                     num_patients: int = 243,
                                     n_samples: int = 1000) -> pd.DataFrame:
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
    subj_wise_agreement = pd.DataFrame(index=gt_series_dict[uncertainties[0]].index,
                                       columns=[f"{uncert}% uncertainty" \
                                       for uncert in uncertainties])
    for uncertainty in uncertainties:
        gt, preds = gt_series_dict[uncertainty], pred_series_dict[uncertainty]
        preds = preds[gt.index] 
        subj_wise_agreement.loc[:, f"{uncertainty}% uncertainty"] = \
            (np.array(gt.values.tolist()) == np.array(preds.values.tolist())).sum(axis=1) / n_samples * 100
    subj_wise_agreement.index.name = "Patient ID"
    return subj_wise_agreement

def calculate_subject_wise_disagreement(*,
                        gt_series_dict: dict[int, pd.Series],
                        pred_series_dict: dict[int, pd.Series],
                        uncertainties: list[int],
                        categories: list[str],
                        num_patients: int = 243,
                        n_samples: int = 1000) -> pd.DataFrame:
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
          columns=[f"{uncert}% uncertainty: % misclassified as {category}" \
            for uncert in uncertainties for category in categories])
    for uncertainty in uncertainties:
        gt, preds = gt_series_dict[uncertainty], pred_series_dict[uncertainty]
        preds = preds[gt.index]
        for i, cat in enumerate(categories):
            subj_wise_disagreement.loc[:, \
            f"{uncertainty}% uncertainty: % misclassified as {cat}"] = \
                np.round((np.array(preds.values.tolist()) == i).sum(axis=1) / n_samples * 100, 2)
        for patient_id in subj_wise_disagreement.index:
            subj_wise_disagreement.loc[patient_id, \
            f"{uncertainty}% uncertainty: % misclassified as {categories[gt[patient_id][0]]}"] = np.nan
    return subj_wise_disagreement
