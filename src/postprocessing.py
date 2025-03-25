"""
Functions for post-processing (visualization and downstream analysis) of simulation results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(cnf_mat: np.ndarray, categories: list[str]) -> None:
    """
    Plot confusion matrix for simulation output.
    """
    if cnf_mat.shape[0] != len(categories):
        raise ValueError("Dimension of confusion matrix does not match the number of categories.")
    cnf_mat_df = pd.DataFrame(cnf_mat.astype(np.int64), index=categories, columns=categories)
    plt.figure(figsize=(10, 10))
    plt.title("Confusion matrix", fontsize=28)
    sns.heatmap(cnf_mat_df, annot=True, cbar=False, fmt="g", annot_kws={"size": 18})
    plt.tick_params(labelsize=18)
    plt.xlabel("Predicted", fontsize=24)
    plt.ylabel("True", fontsize=24)
    plt.show()

def display_differential_classification_results_one_threshold(*, 
                      ad_diff_cls: int, nci_diff_cls: int, num_patients: int) -> None:
    """
    Calculate metrics of differential classification and display results (single threshold).
    """
    print(f"{ad_diff_cls / num_patients * 100:.2f} % simulated subjects were differentially classified"
        " from the Alzheimer's disease category.")
    print(f"{nci_diff_cls / num_patients * 100:.2f} % simulated subjects were differentially classified"
        " from the NCI category.")
    print(f"{(ad_diff_cls + nci_diff_cls) / num_patients * 100:.2f} % simulated subjects were differentially"
        " classified between AD and NCI categories.")
    print(f"Total number of differentially classified individuals: {(ad_diff_cls + nci_diff_cls)}")

def display_differential_classification_results_two_thresholds(*, ad_diff_cls: int, int_diff_cls: int, nci_diff_cls: int, num_patients: int) -> None:
    """
    Calculate metrics of differential classification and display results (two thresholds).
    """
    print(f"{ad_diff_cls / num_patients * 100:.2f} % simulated subjects were differentially classified"
        " from the Alzheimer's disease category.")
    print(f"{int_diff_cls / num_patients * 100:.2f} % simulated subjects were differentially classified"
        " from the intermediate category.")
    print(f"{nci_diff_cls / num_patients * 100:.2f} % simulated subjects were differentially classified"
        " from the NCI category.")
    print("Fraction of simulated subjects differentially classified: Approximately"
        f" {(ad_diff_cls + int_diff_cls + nci_diff_cls) / num_patients * 100:.2f}%")
    print(f"Total number of differentially classified individuals: {(ad_diff_cls + int_diff_cls + nci_diff_cls)}")


def calculate_sens_spec_dual_threshold(cnf_mat: np.ndarray) -> str:
    """
    Calculates and displays sensitivity and specificity for Alzheimer's disease (AD)
    and NCI categories from a confusion matrix for results from dual threshold simulations.
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
                                     gt_arr_dict: dict[int, np.ndarray], 
                                     pred_arr_dict: dict[int, np.ndarray], 
                                     uncertainties: list[int],
                                     num_patients: int = 243, 
                                     n_samples: int = 1000) -> pd.DataFrame:
    """
    Calculate the percent of simulated predictions that agree with the actual prediction for each subject.
    """
    subj_wise_agreement = pd.DataFrame(index=np.arange(num_patients)+1, columns=[f"{uncert}% uncertainty" for uncert in uncertainties])
    for uncertainty in uncertainties:
        gt = gt_arr_dict[uncertainty].reshape(num_patients, n_samples)
        preds = pred_arr_dict[uncertainty].reshape(num_patients, n_samples)
        subj_wise_agreement.loc[:, f"{uncertainty}% uncertainty"] = (gt == preds).sum(axis=1) / n_samples * 100
    subj_wise_agreement.index.name = "Patient ID"
    return subj_wise_agreement

def calculate_subject_wise_disagreement(*,
                                     gt_arr_dict: dict[int, np.ndarray], 
                                     pred_arr_dict: dict[int, np.ndarray], 
                                     uncertainties: list[int],
                                     categories: list[str] = ["NCI", "Intermediate", "AD"],
                                     num_patients: int = 243, 
                                     n_samples: int = 1000) -> pd.DataFrame:
    """
    Calculate the percent of simulated predictions that do not agree with the actual prediction for each subject.
    """
    subj_wise_disagreement = pd.DataFrame(index=np.arange(num_patients)+1, columns=[f"{uncert}% uncertainty: % misclassified as {category}" for uncert in uncertainties for category in categories])
    for uncertainty in uncertainties:
        gt = gt_arr_dict[uncertainty].reshape(num_patients, n_samples)
        preds = pred_arr_dict[uncertainty].reshape(num_patients, n_samples)
        for i, cat in enumerate(categories):
            subj_wise_disagreement.loc[:, f"{uncertainty}% uncertainty: % misclassified as {cat}"] = np.round((preds == i).sum(axis=1) / n_samples * 100, 2)
        for i in range(num_patients):
            subj_wise_disagreement.loc[i+1, f"{uncertainty}% uncertainty: % misclassified as {categories[gt[i, 0]]}"] = np.nan
    subj_wise_disagreement.index.name = "Patient ID"
    return subj_wise_disagreement

