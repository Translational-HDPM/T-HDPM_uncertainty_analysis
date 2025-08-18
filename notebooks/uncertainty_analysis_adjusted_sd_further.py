import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## Modeling of Measurement Uncertainty of a high-dimensional RNA-Seq classifier of cell-free mRNA for Alzheimer’s Disease"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Motivation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Precision Medicine recognizes common complex diseases are actually multiple ‘endotypes’ with different underlying pathology that present with a similar phenotype. Rather than discrete stages of disease, common complex diseases represent a continuum of pathology. End stage common complex disease may not include a complete catalog of biomarkers for earlier points in pathology continuum. Complexity of endotypes presents unique regulatory challenges requiring data simulation to model reproducibility. Translational Diagnostics are transitioning from single analyte assays to multi-analyte, machine learning (ML) classifiers.

    For the use of such 'clinical-grade' ML predictors, it is recommended in the literature to document any sources of variation (aleatoric uncertainty) that affect reproducibility and estimate the variability of the prediction. In this context, methods for estimating ‘clinical-grade’ measurement uncertainty can:

    - improve understanding of test interpretation;
    - act as an operational tool to discern laboratory test drift;
    - shed light on the analytes that should be prioritized to decrease overall uncertainty range and;
    - upon request, help laboratories make estimates of measurement uncertainty available to laboratory users.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Goal""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""This work demonstrates a methodology for estimating the impact of plausible, empirically-informed technical measurement uncertainty on the performance of high-dimensional RNA-Seq classifiers, in benchmark case for Alzheimer's Disease, using Monte Carlo simulations. We simulate the assay variation for each gene independently for a given subject at different levels of relative standard deviation (RSD) of Transcripts Per Million (TPM) values (gene differential expression levels)."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Measurement Uncertainty: Regulatory Setting""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Despite significant advances in diagnostic testing, only a few guidelines have been developed for interpretation of measurement uncertainty in medical laboratories, including

    - Clinical Laboratory Standards Institute (CLSI)
        - CLSI EP29-A Expression of Measurement Uncertainty in Laboratory Medicine
    - International Standards Organization (ISO)
        - ISO 15189-2012
        - ISO/TS 20914:2019
    - Food and Drug Administration (FDA)
        - Class II Special Controls Guidance Document: Ovarian Adnexal Mass Assessment Score Test System (2011).

    Monte Carlo techniques are recommended by FDA to estimate diagnostic uncertainty of multi-dimensional classifiers, which suggest general guidelines through which overall uncertainty of high dimensional classifiers can be determined or estimated. Besides noted variation, sample site, operator and instrument variation need to be considered.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Background""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The ML model used in this notebook is a classifier from a study published in _Science Advances_ ([Toden et al, 2020](https://www.science.org/doi/abs/10.1126/sciadv.abb1654)). The study utilized plasma-derived circulating cell-free messenger RNA (cf-mRNA) from a dataset comprising 126 patients with Alzheimer's Disease (AD) and 116 healthy, non-cognitive impairment (NCI) controls of similar age distribution, sourced from five independent academic centers and one commercial provider; pre-analytic site-specific effects were adjusted for in subsequent analyses. 

    To prepare the machine learning model for AD classification while minimizing bias, feature selection was conducted exclusively on samples from a distinct training cohort (University of Kentucky: 24 NCI and 66 AD patients). This involved identifying differentially expressed genes using DESeq2 with a false discovery rate (FDR) cutoff of less than 0.05.

    The machine learning model, a logistic regression with L2 regularization, was trained using the expression levels (transcripts per million) of these 1658 selected genes from the University of Kentucky training cohort (24 NCI, 66 AD). L2 regularization was specifically employed to prevent overfitting. Metaparameters for this model were optimized using a 15-fold cross-validation strategy on the training cohort. 

    The classifier's ability to discriminate between AD patients and NCI controls was then rigorously evaluated on an independent test set. This test set consisted of the remaining 60 AD patients and 92 NCI controls, derived from four independent sources distinct from the training data (UC San Diego, University of Washington, Indiana University, BioIVT). In this independent validation, the classifier achieved an Area Under the Receiver Operating Characteristic curve (AUC) of 0.83. Further analysis revealed that the genes included in the classifier were enriched in biological pathways known to be associated with AD pathogenesis, including immune response and cellular metabolic processes, thereby lending biological plausibility to the statistical findings.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    return Path, os


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    from src.dtypes import NumpyFloat32Array1D
    from src.logreg_classifier import antilogit_classifier_score
    from src.postprocessing import (
        build_sensitivity_specificity_df,
        calculate_sensitivity_specificity_and_predictive_values,
        calculate_subject_wise_agreement,
        plot_differential_classification_results,
        plot_jaccard_index_plot,
        plot_v_plot,
    )
    from src.simulation import simulate_multiple_uncertainties

    return (
        GridSpec,
        NumpyFloat32Array1D,
        antilogit_classifier_score,
        build_sensitivity_specificity_df,
        calculate_sensitivity_specificity_and_predictive_values,
        calculate_subject_wise_agreement,
        np,
        pd,
        plot_differential_classification_results,
        plot_jaccard_index_plot,
        plot_v_plot,
        plt,
        simulate_multiple_uncertainties,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Setting parameters""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here we set some parameters that we use throughout the analysis.

    1. **Mean TPM**: When we filter genes for analysis with a reduced feature set, we drop genes for which the mean TPM is below this cutoff. (details included later)
    2. **Range of percent uncertainty values to simulate**: List of values taken from literature reported studies representing different overall noise scenario, used as a percentage to scale the baseline technical standard deviation calculated from the TPM value. 
    3. **Number of samples**: Number of Monte Carlo samples to simulate for each subject
    4. **How to aggregate replicates**: How to aggregate TPM values from multiple technical replicates for a given subject. Default is "average", i.e. the average of multiple TPM values will be taken.
    """
    )
    return


@app.cell(hide_code=True)
def _(os):
    master_seed = 123  # Random number seed
    num_parallel_workers = (
        os.cpu_count()
        # Number of parallel jobs to run for simulation
    )
    return master_seed, num_parallel_workers


@app.cell(hide_code=True)
def _(
    mean_tpm_slider,
    n_samples_slider,
    patients_df,
    replicate_aggregation_choice_dropdown,
    uncertainty_range_slider,
):
    n_samples = (
        n_samples_slider.value
    )  # Number of Monte Carlo samples to simulate for each subject
    uncertainties = list(
        range(
            uncertainty_range_slider.value[0],
            uncertainty_range_slider.value[1] + 5,
            5,
        )
    )  # Maximum percentage of noise/uncertainty to simulate. (aka coefficient of variation)
    mean_TPM = (
        mean_tpm_slider.value
    )  # When we filter genes for analysis with a reduced feature set,
    # we drop genes for which the mean TPM is below this cutoff
    collapse_replicates_by = (
        replicate_aggregation_choice_dropdown.value
    )  # How to aggregate TPM values of replicates.
    # Choices include average, maximum and minimum
    num_patients = patients_df.shape[1]
    return (
        collapse_replicates_by,
        mean_TPM,
        n_samples,
        num_patients,
        uncertainties,
    )


@app.cell(hide_code=True)
def _(mo):
    mean_tpm_slider = mo.ui.slider(
        start=0, stop=100, value=0, label="Mean TPM cutoff value", show_value=True
    )
    n_samples_slider = mo.ui.slider(
        start=100,
        stop=10_000,
        step=100,
        value=100,
        label="Number of Monte Carlo samples per TPM value",
        show_value=True,
    )
    uncertainty_range_slider = mo.ui.range_slider(
        start=5,
        stop=40,
        step=1,
        value=[5, 35],
        label="Range of percent uncertainty values to simulate",
        show_value=True,
    )
    replicate_aggregation_choice_dropdown = mo.ui.dropdown(
        ["average", "replicate 1", "replicate 2"],
        value="average",
        label="Collapse replicates by",
    )
    return (
        mean_tpm_slider,
        n_samples_slider,
        replicate_aggregation_choice_dropdown,
        uncertainty_range_slider,
    )


@app.cell(hide_code=True)
def _(
    mean_tpm_slider,
    mo,
    n_samples_slider,
    replicate_aggregation_choice_dropdown,
    uncertainty_range_slider,
):
    mo.callout(
        mo.vstack(
            [
                mean_tpm_slider,
                n_samples_slider,
                uncertainty_range_slider,
                replicate_aggregation_choice_dropdown,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### The Dataset""")
    return


@app.cell(hide_code=True)
def _(mo, patients_df, patients_df_2):
    mo.md(
        rf"""The raw dataset contains gene expression data in the form of TPM values for subjects (including technical replicates), along with the coefficients of the classifier. In addition, we also utilize original disease categories (pathology) that the patients belong to. The dataset contains technical replicates for a subset ({patients_df.shape[1] - patients_df_2.shape[1]} samples) of {patients_df.shape[1]} biological samples. For this analysis, these technical replicates were averaged to create a single expression profile per biological subject. Our subsequent simulation aims to re-introduce a plausible model of this technical variability."""
    )
    return


@app.cell(hide_code=True)
def _(Path, pd):
    raw_data, pathos = None, None
    using_dummy_data = False  # Whether using a dummy dataset
    data_root = Path(__file__).parent.parent.parent / "raw_data"
    if data_root.exists():
        raw_data = pd.read_excel(
            data_root / "ClusterMarkers_1819ADcohort.congregated_DR.xlsx",
            sheet_name=1,
        )
        pathos = pd.read_excel(
            data_root / "ClusterMarkers_1819ADcohort.congregated_DR.xlsx",
            sheet_name=0,
        )
        pathos = pathos.set_index("Isolate ID")
        raw_data = raw_data.set_index("gene_id")
    else:
        # Use dummy data if actual dataset is not available
        using_dummy_data = True
        data_root = Path(__file__).parent.parent / "dummy_data"
        raw_data = pd.read_csv(data_root / "tpm_expression_data.csv")
        pathos = pd.read_csv(data_root / "disease_status_data.csv")
    return pathos, raw_data, using_dummy_data


@app.cell(hide_code=True)
def _(raw_data):
    raw_data.head()
    return


@app.cell(hide_code=True)
def _(pathos):
    pathos.head()
    return


@app.cell(hide_code=True)
def _(pathos):
    pathos_1 = pathos.dropna()
    pathos_1 = pathos_1.loc[pathos_1.index.dropna(), :]
    pathos_1.index = pathos_1.index.astype(int).astype(str)
    return (pathos_1,)


@app.cell(hide_code=True)
def _(np, raw_data):
    patients_df = raw_data[~raw_data.loc[:, "Coeff"].isnull()]
    coefficients = np.nan_to_num(np.array(patients_df.loc[:, "Coeff"]))
    patients_df = patients_df.filter(regex="^\\d+")
    return coefficients, patients_df


@app.cell(hide_code=True)
def _(collapse_replicates_by, pathos_1, patients_df):
    _grouped_cols = patients_df.columns.str.split("-").str[0]
    _grouped = patients_df.T.groupby(_grouped_cols)
    if collapse_replicates_by == "average":
        patients_df_1 = _grouped.apply(lambda x: x.mean()).T
    elif collapse_replicates_by == "replicate 1":
        patients_df_1 = (
            _grouped.apply(lambda x: x[x.index.str.endswith("r1")])
            .reset_index()
            .drop(columns=["level_1"])
            .set_index("level_0")
            .T
        )
    elif collapse_replicates_by == "replicate 2":
        patients_df_1 = (
            _grouped.apply(
                lambda x: x[x.index.str.endswith("r2")]
                if len(x) > 1
                else x[x.index.str.endswith("r1")]
            )
            .reset_index()
            .drop(columns=["level_1"])
            .set_index("level_0")
            .T
        )

    _patients_df_cols_not_in_pathos = []
    for _col in patients_df_1.columns:
        if _col not in pathos_1.index:
            _patients_df_cols_not_in_pathos.append(_col)

    for _col in pathos_1.index:
        if _col not in patients_df_1.columns:
            pathos_2 = pathos_1.drop(_col)

    for _col in _patients_df_cols_not_in_pathos:
        pathos_2.loc[_col] = "NCI"
    patients_df_1 = patients_df_1.loc[:, pathos_2.index]
    return pathos_2, patients_df_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""After pre-processing the matrix (dataframe) of TPM values looks like this:"""
    )
    return


@app.cell(hide_code=True)
def _(patients_df_2):
    patients_df_2.head()
    return


@app.cell(hide_code=True)
def _(mo, patients_df):
    mo.md(
        rf"""Following is a summary of the number of patients in each disease category in the dataset. For each patient, the dataset comprises of TPM values from {patients_df.shape[0]} genes."""
    )
    return


@app.cell(hide_code=True)
def _(pathos_2):
    pathos_2.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Removing genes with low TPM values contributing noise""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""When analysing a dataset with a reduced set of genes, we keep the genes where the mean TPM value is above the specified cutoff. If the `mean_TPM` value is set to zero, no genes are filtered and the entire dataset is used. The objective of removing genes is to filter out any genes with low TPM values that affect the predictions of the classifier by adding noise."""
    )
    return


@app.cell(hide_code=True)
def _(patients_df_1):
    means = patients_df_1.mean(axis=1)
    return (means,)


@app.cell(hide_code=True)
def _(coefficients, mean_TPM, means, num_patients, patients_df_1):
    coefficients_1 = coefficients[means >= mean_TPM]
    patients_df_2 = patients_df_1[means >= mean_TPM]
    patients_df_2 = patients_df_2.iloc[:, :num_patients]
    return coefficients_1, patients_df_2


@app.cell(hide_code=True)
def _(mo, patients_df_1, patients_df_2):
    mo.callout(
        mo.md(
            f"We dropped {patients_df_1.shape[0] - patients_df_2.shape[0]} out of {patients_df_1.shape[0]} genes, i.e. {(patients_df_1.shape[0] - patients_df_2.shape[0]) / patients_df_1.shape[0] * 100.0:.2f}% of the genes."
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""### Selecting the Probability threshold based on Sensitivity and Specificity"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Before simulating uncertainty, we chose to first establish the classifier's decision thresholds on the original, unperturbed data. For deciding on the probability threshold, we use **Youden's J statistic (Youden's index)** which is defined as

    \[
    J = sensitivity + specificity -1
    \]

    where

    \[
    sensitivity = \frac{TP}{TP+FN}
    \]


    \[
    specificity = \frac{TN}{TN+FP}
    \]


    $TP$, $TN$, $FP$ and $FN$ denote the number of true positives, true negatives, false positives and false negatives for a given class, respectively.

    We would like to maximize the Youden's index with our threshold selection. We do this _by selecting a threshold at which sensitivity and specificity are roughly equal_.
    """
    )
    return


@app.cell(hide_code=True)
def _(antilogit_classifier_score, coefficients_1, np, patients_df_2, pd):
    _z_scores = (
        patients_df_2.values - patients_df_2.mean(axis=1).values.reshape(-1, 1)
    ) / patients_df_2.std(axis=1).values.reshape(-1, 1)
    gt_probs = antilogit_classifier_score(
        np.sum(coefficients_1[:, np.newaxis] * _z_scores, axis=0)
    )
    gt_probs = pd.Series(index=patients_df_2.columns, data=gt_probs)
    return (gt_probs,)


@app.cell(hide_code=True)
def _(build_sensitivity_specificity_df, gt_probs, pathos_2):
    ad_sens_spec_df = build_sensitivity_specificity_df(pathos_2, gt_probs, "AD")
    nci_sens_spec_df = build_sensitivity_specificity_df(pathos_2, gt_probs, "NCI")
    return ad_sens_spec_df, nci_sens_spec_df


@app.cell(hide_code=True)
def _(ad_sens_spec_df, nci_sens_spec_df, plt, sns):
    _fig, _ = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18, 6))
    plt.subplot(121)
    sns.lineplot(
        data=ad_sens_spec_df,
        x="threshold",
        y="specificity",
        label="AD specificity",
    )
    sns.lineplot(
        data=ad_sens_spec_df,
        x="threshold",
        y="sensitivity",
        label="AD sensitivity",
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    plt.title("(a) Alzheimer's disease (AD) category")
    plt.subplot(122)
    sns.lineplot(
        data=nci_sens_spec_df,
        x="threshold",
        y="specificity",
        label="NCI specificity",
    )
    sns.lineplot(
        data=nci_sens_spec_df,
        x="threshold",
        y="sensitivity",
        label="NCI sensitivity",
    )
    plt.xlabel("")
    plt.legend()
    plt.title("(b) Non-Cognitively Impaired (NCI) category")
    _fig.text(0.5, 0.05, "Probability threshold", ha="center", va="center")
    _fig.text(0.08, 0.35, "Sensitivity/Specificity", rotation="vertical")
    _fig.suptitle(
        "Figure 1. Sensitivity and specificity versus probability threshold curves for AD and NCI categories."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"Figure 1. Sensitivity and specificity versus probability threshold curves for (a) Alzheimer's disease (AD) category and (b) Non-Cognitively Impaired (NCI) categories. The selection of the probability threshold is done where sensitivity = specificity, or the threshold which maximizes the value of Youden's Index"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    single_threshold_slider_disabled = True

    def set_single_threshold_slider_disabled(arg: bool) -> None:
        global single_threshold_slider_disabled
        single_threshold_slider_disabled = not single_threshold_slider_disabled

    use_youdens_index_for_threshold_switch = mo.ui.switch(
        value=True,
        label="Use Youden's index to set threshold?",
        on_change=set_single_threshold_slider_disabled,
    )
    return (
        single_threshold_slider_disabled,
        use_youdens_index_for_threshold_switch,
    )


@app.cell(hide_code=True)
def _(
    mo,
    single_threshold_slider_disabled,
    use_youdens_index_for_threshold_switch,
):
    single_threshold_slider = mo.ui.slider(
        start=0.01,
        stop=0.99,
        step=0.01,
        value=0.03,
        label="Single threshold value",
        show_value=True,
        disabled=single_threshold_slider_disabled,
    )

    _elements = [use_youdens_index_for_threshold_switch, single_threshold_slider]
    mo.callout(mo.vstack(_elements))
    return (single_threshold_slider,)


@app.cell(hide_code=True)
def _(
    ad_sens_spec_df,
    mo,
    np,
    single_threshold_slider,
    use_youdens_index_for_threshold_switch,
    using_dummy_data,
):
    single_thres = 0.03
    if use_youdens_index_for_threshold_switch.value:
        _precision = 0 if using_dummy_data else 2
        _rounded_sens = ad_sens_spec_df["sensitivity"].apply(
            lambda x: np.round(x, _precision)
        )
        _rounded_spec = ad_sens_spec_df["specificity"].apply(
            lambda x: np.round(x, _precision)
        )
        _filt = _rounded_sens == _rounded_spec
        mo.stop(
            _filt.sum() < 1,
            mo.callout(
                mo.md(
                    """///warning
                    A probability threshold value maximizing Youden's index for the current mean TPM cutoff value was not found. Change the mean TPM cutoff value to continue."""
                )
            ),
        )

        _target_ad_specificity = (
            ad_sens_spec_df.loc[_filt, "specificity"].values[0] * 100
        )  # For single threshold, roughly where sensitivity = specificity
        single_thres = ad_sens_spec_df.loc[_filt, "threshold"].values[0] / 100
        single_thres = max(0.03, single_thres)  # Set a floor of 0.03 for the threshold
        # print(f"{single_thres=}")
    else:
        single_thres = single_threshold_slider.value
    return (single_thres,)


@app.cell(hide_code=True)
def _(
    calculate_sensitivity_specificity_and_predictive_values,
    gt_probs,
    pathos_2,
    single_thres,
):
    ad_sens, ad_spec, ad_ppv, ad_npv = (
        calculate_sensitivity_specificity_and_predictive_values(
            pathos_2["Disease"].apply(lambda x: 1 if x == "AD" else 0),
            gt_probs.apply(lambda x: 1 if x >= single_thres else 0),
            0,
        )
    )
    ad_spec = ad_spec * 100.0
    ad_sens = ad_sens * 100.0
    ad_ppv *= 100.0
    ad_npv *= 100.0
    return ad_npv, ad_ppv, ad_sens, ad_spec


@app.cell(hide_code=True)
def _(ad_npv, ad_ppv, ad_sens, ad_spec, mo, single_thres):
    mo.md(
        rf"""
    At threshold {single_thres:.4f}, for AD,

    - sensitivity = {ad_sens:.4f},
    - specificity = {ad_spec:.4f},
    - positive predictive value = {ad_ppv:.4f},
    - negative predictive value = {ad_npv:.4f},
    - Youden's index = {(ad_sens + ad_spec - 100) / 100:.4f}.

    So, we use {single_thres:.4f} as our probability threshold for the single threshold classification scenario.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, single_thres):
    mo.md(
        rf"""For two thresholds, we find lower and upper thresholds that maximize Youden's index for NCI and AD classes respectively. However, we are limited by our dataset, since the diagnoses are dichotomised. If there were a third intermediate category between AD and NCI, we could have calculated distinct lower and upper thresholds from the sensitivity and specificity information. But if we try to find lower and upper thresholds following the criteria stated before, we end up with the same lower and upper threshold. To mitigate this we manually set the lower threshold equal to the single threshold ({single_thres:.4f}) as calculated before, and upper threshold at few steps from the lower threshold."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    threshold_step_slider = mo.ui.slider(
        start=0.01,
        stop=1.00,
        step=0.005,
        value=0.06,
        show_value=True,
        label="Step to set the upper threshold away from the lower threshold.",
    )
    mo.callout(threshold_step_slider)
    return (threshold_step_slider,)


@app.cell(hide_code=True)
def _(single_thres, threshold_step_slider):
    dual_thres_low = single_thres
    dual_thres_high = min(1.0, single_thres + threshold_step_slider.value)
    return dual_thres_high, dual_thres_low


@app.cell(hide_code=True)
def _(
    calculate_sensitivity_specificity_and_predictive_values,
    dual_thres_high,
    dual_thres_low,
    gt_probs,
    pathos_2,
):
    nci_sens_low, nci_spec_low, nci_ppv_low, nci_npv_low = (
        calculate_sensitivity_specificity_and_predictive_values(
            pathos_2["Disease"].apply(lambda x: 1 if x == "AD" else 0),
            gt_probs.apply(lambda x: 1 if x >= dual_thres_low else 0),
            0,
        )
    )
    ad_sens_high, ad_spec_high, ad_ppv_high, ad_npv_high = (
        calculate_sensitivity_specificity_and_predictive_values(
            pathos_2["Disease"].apply(lambda x: 1 if x == "AD" else 0),
            gt_probs.apply(lambda x: 1 if x >= dual_thres_high else 0),
            1,
        )
    )
    ad_spec_high = ad_spec_high * 100.0
    ad_sens_high = ad_sens_high * 100.0
    ad_ppv_high *= 100.0
    ad_npv_high *= 100.0
    nci_spec_low = nci_spec_low * 100.0
    nci_sens_low = nci_sens_low * 100.0
    nci_ppv_low *= 100.0
    nci_npv_low *= 100.0
    return (
        ad_sens_high,
        ad_spec_high,
        nci_npv_low,
        nci_ppv_low,
        nci_sens_low,
        nci_spec_low,
    )


@app.cell(hide_code=True)
def _(
    ad_sens_high,
    ad_spec_high,
    dual_thres_high,
    dual_thres_low,
    mo,
    nci_npv_low,
    nci_ppv_low,
    nci_sens_low,
    nci_spec_low,
):
    mo.md(
        rf"""
    At lower threshold {dual_thres_low:.4f}, for NCI,

    - sensitivity = {nci_sens_low:.4f}
    - specificity = {nci_spec_low:.4f}
    - positive predictive value = {nci_ppv_low:.4f}
    - negative predictive value = {nci_npv_low:.4f}
    - Youden's index = {(nci_sens_low + nci_spec_low - 100) / 100:.4f}.

    At upper threshold {dual_thres_high:.4f}, for AD

    - sensitivity = {ad_sens_high:.4f}
    - specificity = {ad_spec_high:.4f}
    - positive predictive value = {nci_ppv_low:.4f}
    - negative predictive value = {nci_npv_low:.4f}
    - Youden's index = {(ad_sens_high + ad_spec_high - 100) / 100:.4f}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Performing Monte Carlo simulations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We follow established guidance by the FDA ([Ovarian Adnexal Mass Assessment Score system (2011)](https://www.fda.gov/medical-devices/guidance-documents-medical-devices-and-radiation-emitting-products/ovarian-adnexal-mass-assessment-score-test-system-class-ii-special-controls-guidance-industry-and)) in simulating technical variation in the TPM values. The specific steps as mentioned in the guidance are as follows (for sake of simplicity, consider two individual analytes $X_1$ and $X_2$ with $Score = F(X_1, X_2)$ and repeatability precision data):

    1. Provide repeatability precision results (mean value, standard deviation (SD), and percentage coefficient of variation (%CV)) from previously performed precision studies and from the precision studies for the Score. Using these data, construct repeatability precision profiles for $X_1$ and $X_2$ by linear interpolation.

    2. Consider a combination of two analytes with values $X_1 = U$ and $X_2 = V$. Using repeatability precision profiles, obtain $SD_1 (U)$ for $X_1 = U$ and $SD_2 (V)$ for $X_2 = V$.

    3. Generate $X_1^*$ using normal distribution with mean value of $U$ and standard deviation of $SD_1 (U)$ and generate $X_2^*$ using normal distribution with mean value of $V$ and standard deviation of $SD_2 (V)$. Calculate $Score^* = F(X_1^*, X_2^*)$. After performing this step $K$ times (for example, 100), calculate the mean value of score of $K$ measurements $Score^*_{mean}$ (corresponding to mean value of the score for $X_1 = U$ and $X_2 = V$) and standard deviation SD and %CV of the $K$ score measurements.

    4. Provide repeatability precision profile for the Score: values of the mean score $Score^* _{mean}$ with the SD and %CV from the previous step for all possible combinations of $U$ and $V$ for which precision profiles are available. 

    As described in the guidance above, given a measurement $\mu$, to simulate $k$ % uncertainty ($k$% coefficient of variation/relative standard deviation), we sample from a Gaussian distribution with mean $\mu$ and standard deviation (SD) $kX/100$. However, for RNA-seq datasets, modeling uncertainty in this fashion with a constant noise level ignores the trend of technical variation commonly observed in which the standard deviation (technical variation) decreases with increasing TPM values to an asymptote. (e.g. in Fig. 1.(a) from [Law et al (2014)](https://link.springer.com/content/pdf/10.1186/gb-2014-15-2-r29.pdf)).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Aligning with Law et al (2014), we adopt the following functional form to model the relationship between the standard deviation and the mean

    $$
    \sigma = \frac{a}{b + \mu} + c
    $$

    where $\mu$ and $\sigma$ are mean and standard deviation of the $log_2 (1+TPM)$ dataset, respectively and $a$, $b$, $c$ are constants.

    We start with values of $a$ = 0.75, $b$ = 1.0, $c$ = 0.25, $scaling factor$ = 6.0. 

    ///note
    The values of $a$, $b$, $c$ and $scaling factor$ have been set without empirical calculations due to lack of sufficient technical replicate data. Ideally, if technical replicate data is available, these parameters should be set empirically.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To generate Monte Carlo TPM samples for a given gene $i$ and subject $j$, we follow the following algorithm:

    1. Calculate $\sigma_{ij}$ using the SD-mean relationship with $TPM_{ij}$ as the mean:

    $$\mu_{ij} = log_{2}(1 + TPM_{ij})$$

    $$\sigma_{ij} = \frac{a}{\mu_{ij} + b} + c$$

    $$\sigma_{ij, scaled} = \frac{scaling factor * k * \sigma_{ij}}{100}$$

    2. Generate $N$ samples $x_k$ ($k$ = $1…N$) from a Gaussian distribution with mean = $\mu_{ij}$ and standard deviation = $\sigma_{ij}$, scaled.
    3. Convert samples to TPM scale by exponentiation:
    $$TPM_{k, sampled} = 2^{x_k}$$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def calculate_scaled_sd(
        tpm: float,
        uncertainty_pct: int | float,
        a: float = 0.75,
        b: float = 1.0,
        c: float = 0.25,
        scaling_factor: float = 6.0,
    ) -> float:
        r"""
        Calculate scaled standard deviation for a given TPM value and baseline
        uncertainty to simulate based on the equation
        $$
        \sigma = \frac{a}{b + \mu} + c
        $$

        Parameters
        ----------
        tpm
            The TPM value to use as the input.
        uncertainty_pct
            The baseline percent uncertainty.
        a
            Constant for the calculation.
        b
            Constant for the calculation.
        c
            Constant for the calculation.
        scaling_factor
            Multiplier used to scale the SD from the equation.

        Returns
        -------
        float
            The scaled standard deviation value to use for generating simulated TPM
            values.
        """
        sigma = a / (np.log2(tpm + 1) + b) + c
        scaled_pct_sd = scaling_factor * uncertainty_pct * sigma
        return scaled_pct_sd / 100

    return (calculate_scaled_sd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    After generating the simulated measurements, we calculate the classifier predictions for these. We then compare the score for the simulated measurement ("simulated score") against the score for the actual measurement ("inferent score"). Based on the set decision threshold(s) for the classifier, we track the classification of the simulated scores against the inferent scores.

    We define a **differentially classified subject** as a subject (patient) that satisfies a differential classification criterion to produce a different classification into AD/NCI than the inferent category. We track differentially classified subjects under the differential classification scenario where at least 10 % of the simulated scores are different from the actual score for a given subject.
    """
    )
    return


@app.cell(hide_code=True)
def _(NumpyFloat32Array1D, calculate_scaled_sd, np):
    def sampler(
        tpm: float,
        baseline_rsd: float,
        n_points: int = 1000,
        seed: int | None = None,
    ) -> NumpyFloat32Array1D:
        """
        Function to generate Monte Carlo TPM samples given a TPM value and a
        baseline uncertainty value.

        Parameters
        ----------
        tpm
            TPM value to generate simulated TPM values from.
        baseline_rsd
            Reference uncertainty value to calculate the scaled SD for the simulated
            TPM values. Must be between 0 and 1.
        n_points
            Number of simulated TPM values to generate. Defaults to 1000.
        seed
            Seed for random number generator. Default is None.

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.float32]]
            1D numpy array of floating point values representing TPM samples.
        """
        rng = np.random.default_rng(seed)
        scaled_sd = calculate_scaled_sd(tpm, baseline_rsd * 100)
        return np.pow(2.0, rng.normal(np.log2(tpm + 1), scaled_sd, n_points))

    return (sampler,)


@app.cell(hide_code=True)
def _(
    coefficients_1,
    dual_thres_high,
    dual_thres_low,
    master_seed,
    n_samples,
    num_parallel_workers,
    patients_df_2,
    sampler,
    simulate_multiple_uncertainties,
    single_thres,
    uncertainties,
):
    res = simulate_multiple_uncertainties(
        patients_df_2,
        sampler,
        uncertainties,
        thres_low=dual_thres_low,
        thres_high=dual_thres_high,
        single_thres=single_thres,
        coefficients=coefficients_1,
        diff_class_lim=int(0.1 * n_samples),
        n_samples=n_samples,
        seed=master_seed,
        num_workers=num_parallel_workers,
    )
    return (res,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualization and post-processing""")
    return


@app.cell(hide_code=True)
def _(
    ad_spec,
    ad_spec_high,
    nci_spec_low,
    plot_differential_classification_results,
    res,
    uncertainties,
):
    plot_differential_classification_results(
        labels_dict_single_thres={"NCI": "b", "AD": "g"},
        labels_dict_dual_thres={"NCI": "b", "Intermediate": "r", "AD": "g"},
        gt_labels_single_thres=res.single_thres_gt_labels,
        gt_labels_dual_thres=res.dual_thres_gt_labels,
        pred_labels_dict_single_thres=res.single_thres_pred_labels,
        pred_labels_dict_dual_thres=res.dual_thres_pred_labels,
        single_thres_plot_title=f"(a) Single threshold, specificity: {ad_spec:.2f} % AD",
        dual_thres_plot_title=f"(b) Dual threshold, specificities: {nci_spec_low:.2f}% NCI and {ad_spec_high:.2f}% AD",
        figure_title="Figure 2. Differential classification for levels of uncertainty"
        + f" ({min(uncertainties)}-{max(uncertainties)}%)\n"
        + "(at least 10% simulated scores mismatch)",
    )
    return


@app.cell(hide_code=True)
def _(mo, uncertainties):
    mo.callout(
        mo.md(rf"""
        Figure 2. Differential classification for levels of uncertainty  ({min(uncertainties)}-{max(uncertainties)}%) (at least 10% simulated scores mismatch).  As simulated measurement uncertainty increases, a larger percentage of both AD and NCI subjects experience a change in their diagnostic classification in at least 10% of the simulated scenarios.
    """)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Jaccard index

    To analyze the impact of simulated technical variation on the classifier's predictions, we use the Jaccard index to compare the predicted categories under synthetic noise against predicted categories with no perturbations. The Jaccard index calculates how similar two finite sets are. For two sets $A$ and $B$, it is defined as the ratio of the size of their intersection to the size of their union.

    $$
    J(A, B) = \frac{|A \cap B|}{|A \cup B|}
    $$

    If $A$ and $B$ are identical, the Jaccard index is 1. In this context, the lower the Jaccard index, the more dissimilar the predictions under noise are from the predictions without noise, indicating increased differential classification.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    ad_spec,
    ad_spec_high,
    nci_spec_low,
    plot_jaccard_index_plot,
    res,
    uncertainties,
):
    plot_jaccard_index_plot(
        labels_dict_single_thres={"NCI": "b", "AD": "g"},
        labels_dict_dual_thres={"NCI": "b", "Intermediate": "r", "AD": "g"},
        gt_labels_single_thres=res.single_thres_gt_labels,
        gt_labels_dual_thres=res.dual_thres_gt_labels,
        pred_labels_dict_single_thres=res.single_thres_pred_labels,
        pred_labels_dict_dual_thres=res.dual_thres_pred_labels,
        single_thres_plot_title=f"(a) Single threshold, specificity: {ad_spec:.2f} % AD",
        dual_thres_plot_title=f"(b) Dual threshold, specificities: {nci_spec_low:.2f}% NCI and {ad_spec_high:.2f}% AD",
        figure_title="Figure 3. Jaccard index plot showing differential classification"
        + f" for levels of uncertainty ({min(uncertainties)}-{max(uncertainties)} %) \n"
        + "(at least 10% simulated scores mismatch)",
    )
    return


@app.cell(hide_code=True)
def _(mo, uncertainties):
    mo.callout(
        mo.md(rf"""
    Figure 3. Jaccard index plot showing differential classification for levels of uncertainty ({min(uncertainties)}-{max(uncertainties)} %) (at least 10% simulated scores mismatch). The Intermediate group's classification is sensitive to measurement noise, leading to 100% reclassification ($J = 0$) when uncertainty reaches approximately ~15%. This highlights the instability in classification for the intermediate category due to proximity to decision boundaries.

    """)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualizing agreement between simulated and inferent scores""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We calculate the classification agreement between methods in each part of the spectrum of disease severity using the **V-plot method** ([Petraco et al (2018)](https://openheart.bmj.com/content/openhrt/5/1/e000663.full.pdf)). 

    The V-plot has this shape because the accuracy of tests is universally high at the extremes of disease severity (near 100%) but close to the classification cut-off agreement plunges. The width of the mouth of the V can be used as a general measure of a test’s performance: the wider the V, the poorer the test ability to match a reference modality. Classification agreement between two methods of measurement is called diagnostic accuracy if one test is considered the reference gold standard.
    """
    )
    return


@app.cell(hide_code=True)
def _(calculate_subject_wise_agreement, n_samples, res, uncertainties):
    single_thres_subj_wise_agreement = calculate_subject_wise_agreement(
        gt_series_dict=res.single_thres_gt_series,
        pred_series_dict=res.single_thres_pred_series,
        uncertainties=uncertainties,
        n_samples=n_samples,
    )
    dual_thres_subj_wise_agreement = calculate_subject_wise_agreement(
        gt_series_dict=res.dual_thres_gt_series,
        pred_series_dict=res.dual_thres_pred_series,
        uncertainties=uncertainties,
        n_samples=n_samples,
    )
    return dual_thres_subj_wise_agreement, single_thres_subj_wise_agreement


@app.cell(hide_code=True)
def _(
    GridSpec,
    ad_spec,
    ad_spec_high,
    dual_thres_subj_wise_agreement,
    gt_probs,
    nci_spec_low,
    pathos_2,
    plot_v_plot,
    plt,
    single_thres_subj_wise_agreement,
    sns,
    uncertainties,
):
    ad_probs = gt_probs[pathos_2[pathos_2["Disease"] == "AD"].index]
    nci_probs = gt_probs[pathos_2[pathos_2["Disease"] == "NCI"].index]
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    fig.add_subplot(gs[0])
    plot_v_plot(
        single_thres_subj_wise_agreement,
        gt_probs,
        uncertainties,
        f"(a) Single threshold, specificity: {ad_spec:.2f}% AD",
        False,
        False,
    )
    plt.xlim([0.0, 1.0])
    plt.ylabel("Percent agreement between simulated and\n inferent scores for subjects")
    plt.gca().set_xticklabels([])
    fig.add_subplot(gs[1])
    plot_v_plot(
        dual_thres_subj_wise_agreement,
        gt_probs,
        uncertainties,
        f"(b) Dual threshold, specificities: {nci_spec_low:.2f}% NCI and {ad_spec_high:.2f}% AD",
        False,
        False,
    )
    plt.xlim([0.0, 1.0])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.ylabel("")
    leg_handles, leg_labels = plt.gca().get_legend_handles_labels()
    for i in [3, 4]:
        fig.add_subplot(gs[i - 1])
        sns.histplot(
            ad_probs,
            color="r",
            bins=30,
            label="Classifier probability score for AD patients",
            fill=True,
            alpha=0.3,
        )
        sns.histplot(
            nci_probs,
            color="b",
            bins=30,
            label="Classifier probability score for NCI patients",
            fill=True,
            alpha=0.3,
        )
        plt.xlim([0.0, 1.0])
        if i == 4:
            plt.gca().set_yticklabels([])
            plt.ylabel("")
    leg_handles_2, leg_labels_2 = plt.gca().get_legend_handles_labels()
    fig.legend(
        leg_handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(uncertainties) // 2,
    )
    fig.legend(
        leg_handles_2,
        leg_labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=len(uncertainties) // 2,
    )
    fig.text(0.5, 0.07, "Probability score", va="center", ha="center")
    fig.suptitle(
        "Figure 4. V-plot showing agreement between simulated and inferent scores"
        + f" for levels of uncertainty ({min(uncertainties)}-{max(uncertainties)}%)",
        fontsize=14,
    )
    fig.text(
        0.1,
        -0.15,
        "*The histograms below the v-plots show the distribution of classifier probability scores",
        ha="left",
        va="center",
    )
    return


@app.cell(hide_code=True)
def _(mo, uncertainties):
    mo.callout(
        mo.md(rf"""Figure 4. V-plot showing agreement between simulated and inferent scores for levels of uncertainty ({min(uncertainties)}-{max(uncertainties)}%). The agreement between inferent and simulated scores drops as we move closer to the threshold. The “V” near the threshold gets wider with increased simulated uncertainty level. 
    """)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""#### Which subjects flipped their classes (categories) under simulated noise?"""
    )
    return


@app.cell(hide_code=True)
def _(mo, uncertainties, uncertainty_range_slider):
    uncertainty_slider = mo.ui.slider(
        start=uncertainties[0],
        stop=uncertainties[-1],
        step=uncertainty_range_slider.step,
        value=uncertainties[-1],
        show_value=True,
        label="Select uncertainty level",
    )
    mo.callout(uncertainty_slider)
    return (uncertainty_slider,)


@app.cell(hide_code=True)
def _(mo, pd, res, uncertainty_slider):
    _uncert = uncertainty_slider.value
    _single_thres_label_dict = {0: "NCI", 1: "AD"}
    _dual_thres_label_dict = {0: "NCI", 1: "Intermediate", 2: "AD"}

    def get_display_df(
        gt_labels: pd.Series,
        pred_labels: pd.Series,
        label_dict: dict[int, str],
    ) -> pd.DataFrame:
        display_df = pd.DataFrame(
            [
                gt_labels.apply(lambda x: label_dict[x]),
                pred_labels.apply(lambda x: label_dict[x]),
            ]
        ).T
        display_df.rename(columns={0: "No noise", 1: "With noise"}, inplace=True)
        display_df.index.name = "Patient ID"
        display_df["Category changed"] = (
            display_df["No noise"] != display_df["With noise"]
        )
        return display_df

    _tabs = mo.ui.tabs(
        {
            "Single threshold": get_display_df(
                res.single_thres_gt_labels,
                res.single_thres_pred_labels[_uncert],
                _single_thres_label_dict,
            ),
            "Dual threshold": get_display_df(
                res.dual_thres_gt_labels,
                res.dual_thres_pred_labels[_uncert],
                _dual_thres_label_dict,
            ),
        },
        value="Subject-wise categories with and without simulated noise",
    )
    _tabs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "To observe the effects of filtering out genes with low mean TPM, please change the value in the interactive mean TPM cutoff slider. The plots will be regenerated automatically."
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preliminary observations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - RNA-Seq Measurement Uncertainty impacts differential classification predominantly at the classifier threshold. (Ref. Figure 4)
    - Filtering out genes with low mean TPMs decreases the percentage of both AD and NCI subjects whose diagnostic classification changes when at least 10% of simulations mismatch. (Ref. Figure 2 and 3)
    - These low-expression genes are a source of classification instability under uncertainty which need to be taken into account when building diagnostic classifications. (Ref. Figure 4)
    - For a diagnostic classifier with two thresholds, (Ref. Figure 4) 
        - An initial lab result might suggest a wait and watch approach.
        - The demonstration of near-certain reclassification under typical measurement noise for this specific patient's result acts like a diagnostic stress test (i.e revealing high instability).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Additional analysis on the Low TPM genes""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ///note
    Set the "Mean TPM cutoff value" to a non-zero number to see the effect on the cells below.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Distribution of TPM values

    We want to check the distribution of simulated TPM values for low TPM genes in patients diagnosed as NCI.
    """
    )
    return


@app.cell(hide_code=True)
def _(coefficients, mean_TPM, means, pathos_2, patients_df_1, patients_df_2):
    low_tpm_patients_df = patients_df_1[means < mean_TPM]
    low_tpm_coefficients = coefficients[means < mean_TPM]
    _ad_patients = (pathos_2[pathos_2["Disease"] == "AD"]).index
    low_tpm_patients_df = low_tpm_patients_df.drop(
        columns=list(_ad_patients)
    )  # Keep only NCI subjects but low TPM genes
    full_set_patients_df = patients_df_1.drop(
        columns=list(_ad_patients)
    )  # Keep only NCI subjects but all genes
    filtered_set_patients_df = patients_df_2.drop(
        columns=list(_ad_patients)
    )  # Keep only NCI subjects but high TPM genes
    return (
        filtered_set_patients_df,
        full_set_patients_df,
        low_tpm_coefficients,
        low_tpm_patients_df,
    )


@app.cell(hide_code=True)
def _(NumpyFloat32Array1D, np, pd):
    from typing import Callable

    from matplotlib.figure import Figure

    from src.simulation import simulate_sampling_experiment

    def generate_samples(
        tpm_df: pd.DataFrame,
        sampler: Callable[[float, float, int], NumpyFloat32Array1D],
        uncertainty: int,
        n_samples: int,
        seed: int,
    ) -> NumpyFloat32Array1D:
        n_features, num_patients = tpm_df.shape[0], tpm_df.shape[1]
        # Generate random number seed sequence for seeds for sampler
        seed_seq = np.random.SeedSequence([654, seed])
        all_samples = []
        for j in range(num_patients):
            samples = np.zeros((n_features, n_samples))

            # Spawn n_feature seeds, one seed per feature
            seeds = seed_seq.spawn(n_features)
            for i in range(n_features):
                mean = tpm_df.iloc[i, j]

                # Generate Monte Carlo samples
                samples[i] = sampler(mean, uncertainty / 100, n_samples, seeds[i])
            all_samples.append(samples)
        return np.hstack(all_samples).ravel()

    return Figure, generate_samples, simulate_sampling_experiment


@app.cell(hide_code=True)
def _(
    antilogit_classifier_score,
    low_tpm_coefficients,
    low_tpm_patients_df,
    np,
    pd,
):
    _z_scores = (
        low_tpm_patients_df.values
        - low_tpm_patients_df.mean(axis=1).values.reshape(-1, 1)
    ) / low_tpm_patients_df.std(axis=1).values.reshape(-1, 1)
    low_tpm_gt_probs = antilogit_classifier_score(
        np.sum(low_tpm_coefficients[:, np.newaxis] * _z_scores, axis=0)
    )
    low_tpm_gt_probs = pd.Series(
        index=low_tpm_patients_df.columns, data=low_tpm_gt_probs
    )
    return


@app.cell(hide_code=True)
def _(Figure, NumpyFloat32Array1D, plt, sns):
    def plot_histogram(
        data_1: NumpyFloat32Array1D,
        data_2: NumpyFloat32Array1D,
        *,
        xlabel: str,
        label_1: str,
        label_2: str,
        fig_title: str,
        uncertainty: int,
        nbins: int | None = 30,
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(
            data_1,
            color="b",
            alpha=0.3,
            fill=True,
            bins=nbins,
            stat="density",
            label=label_1,
            ax=ax,
        )
        sns.histplot(
            data_2,
            color="r",
            alpha=0.3,
            fill=True,
            bins=nbins,
            stat="density",
            label=label_2,
            ax=ax,
        )
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend(loc="best")
        plt.title(
            f"Histogram of {fig_title} from \n{label_1} and {label_2} at {uncertainty}% simulated uncertainty."
        )
        return fig

    return (plot_histogram,)


@app.cell(hide_code=True)
def _(
    generate_samples,
    low_tpm_patients_df,
    master_seed,
    n_samples,
    plt,
    sampler,
    sns,
    uncertainties,
):
    _uncert = uncertainties[-1]
    low_tpm_patients_samples = generate_samples(
        low_tpm_patients_df, sampler, uncertainties[-1], n_samples, master_seed
    )
    _nbins = 30
    _fig, _ax = plt.subplots(figsize=(12, 8))
    sns.histplot(
        low_tpm_patients_samples,
        color="b",
        alpha=0.3,
        fill=True,
        bins=_nbins,
        stat="density",
        ax=_ax,
    )
    plt.xlabel("TPM values")
    plt.ylabel("Density")
    plt.title("Histogram of simulated TPM values from low TPM genes of NCI patients")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Does removing low TPM genes increase differential classification of NCI subjects?

    We want to test the hypothesis that removing them causes increased differential classification of patients in the NCI category.
    """
    )
    return


@app.cell(hide_code=True)
def _(
    coefficients,
    coefficients_1,
    dual_thres_high,
    dual_thres_low,
    filtered_set_patients_df,
    full_set_patients_df,
    master_seed,
    n_samples,
    plot_histogram,
    sampler,
    simulate_sampling_experiment,
    single_thres,
    uncertainties,
):
    _uncert = uncertainties[-1]
    (
        _,
        _,
        unfiltered_neg_subscores,
        unfiltered_pos_subscores,
        _,
        unfiltered_probs,
    ) = simulate_sampling_experiment(
        full_set_patients_df,
        sampler,
        dual_thres_1=dual_thres_low,
        dual_thres_2=dual_thres_high,
        single_thres=single_thres,
        diff_class_lim=int(0.1 * n_samples),
        uncertainty=_uncert,
        n_samples=n_samples,
        coefficients=coefficients,
        seed=master_seed,
    )
    _, _, filtered_neg_subscores, filtered_pos_subscores, _, filtered_probs = (
        simulate_sampling_experiment(
            filtered_set_patients_df,
            sampler,
            dual_thres_1=dual_thres_low,
            dual_thres_2=dual_thres_high,
            single_thres=single_thres,
            diff_class_lim=int(0.1 * n_samples),
            uncertainty=_uncert,
            n_samples=n_samples,
            coefficients=coefficients_1,
            seed=master_seed,
        )
    )

    plot_histogram(
        filtered_probs,
        unfiltered_probs,
        xlabel="Classifier scores (probabilities)",
        label_1="Filtered dataset",
        label_2="Unfiltered dataset",
        fig_title="classifier scores",
        uncertainty=_uncert,
    )
    return (
        filtered_neg_subscores,
        filtered_pos_subscores,
        unfiltered_neg_subscores,
        unfiltered_pos_subscores,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### What percent of TPM values in low TPM genes among NCI subjects are "Null" values?

    In the original dataset, the TPM values that were missing must have been imputed with 0 to avoid calculation errors. Since the present dataset actually does not have null values, but while generation had all TPM values less than 5 removed, we *assume* that **all TPM values that are zero were previously missing or were considered insignificant**. Getting the proportion of such values for every low TPM gene within NCI patients will help us pinpoint the degree of noise they add to the data.
    """
    )
    return


@app.cell(hide_code=True)
def _(low_tpm_patients_df):
    _percent_missing_low_tpm_genes = (
        low_tpm_patients_df.apply(lambda x: x == 0).sum(axis=1)
        / low_tpm_patients_df.shape[1]
        * 100
    )  # Convert to %
    _percent_missing_low_tpm_genes.name = "Percent missing values from NCI subjects"
    _percent_missing_low_tpm_genes.index.name = "gene"
    _percent_missing_low_tpm_genes.sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### How many of the low TPM genes had positive coefficients and negative coefficients?

    We want to see if removing the low TPM genes removes an equal number of genes with positive and negative coefficients.
    """
    )
    return


@app.cell(hide_code=True)
def _(low_tpm_coefficients, mo):
    mo.md(
        rf"""
    - Number of negative coefficient genes = {(low_tpm_coefficients < 0).sum()}
    - Number of positive coefficient genes = {(low_tpm_coefficients >= 0).sum()}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""#### Did filtering out low TPM genes reduce the contribution of "resilience" factors to the aggregated classifier score?"""
    )
    return


@app.cell(hide_code=True)
def _(
    filtered_neg_subscores,
    filtered_pos_subscores,
    mo,
    np,
    plot_histogram,
    uncertainties,
    unfiltered_neg_subscores,
    unfiltered_pos_subscores,
):
    _uncert = uncertainties[-1]
    _unfiltered_neg_subscore_fracs = np.abs(unfiltered_neg_subscores) / (
        np.abs(unfiltered_neg_subscores) + np.abs(unfiltered_pos_subscores)
    )
    _filtered_neg_subscore_fracs = np.abs(filtered_neg_subscores) / (
        np.abs(filtered_neg_subscores) + np.abs(filtered_pos_subscores)
    )

    _tabs = mo.ui.tabs(
        {
            "Positive subscores": plot_histogram(
                1 - _filtered_neg_subscore_fracs,
                1 - _unfiltered_neg_subscore_fracs,
                xlabel="Classifier positive subscores fraction (contribution from genes with positive coefficients)",
                label_1="Filtered dataset",
                label_2="Unfiltered dataset",
                fig_title="classifier positive subscores fraction",
                uncertainty=_uncert,
            ),
            "Negative subscores": plot_histogram(
                _filtered_neg_subscore_fracs,
                _unfiltered_neg_subscore_fracs,
                xlabel="Classifier negative subscores fraction (contribution from genes with negative coefficients)",
                label_1="Filtered dataset",
                label_2="Unfiltered dataset",
                fig_title="classifier negative subscores fraction",
                uncertainty=_uncert,
            ),
        },
        value="Effect of filtering on subscore contributions to classifier score",
    )
    _tabs
    return


@app.cell(hide_code=True)
def _(gt_probs):
    gt_probs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### References""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. Beaver et al. "An FDA Perspective on the Regulatory Implications of Complex Signatures to Predict Response to Targeted Therapies." _Clin Cancer Res. 2017 Mar 15;23(6):1368-1372._
    2. Braga and Panteghini "The utility of measurement uncertainty in medical laboratories" _Clin Chem Lab Med 2020; 58(9):1407-1413_.
    3. Law, Charity W., et al. "voom: Precision weights unlock linear model analysis tools for RNA-seq read counts." _Genome biology 15 (2014): 1-17_.
    4. Petraco, Ricardo, et al. "Effects of disease severity distribution on the performance of quantitative diagnostic methods and proposal of a novel ‘V-plot’methodology to display accuracy values." _Open Heart 5.1 (2018): e000663_.
    5. Plebani et al. "Measurement uncertainty: light in the shadows" _Clin Chem Lab Med 2020; 58(9):1381-1383_.
    6. Song et al. "Anvil – System Architecture and Experiences from Deployment and Early User Operations" _PEARC July (2022)_.
    7. Theodorsson E. "Uncertainty in Measurement and Total Error: Tools for Coping with Diagnostic Uncertainty." _Clin Lab Med. 2017 Mar;37(1):15-34_.
    8. Toden et al. "Noninvasive characterization of Alzheimer's disease by circulating, cell-free messenger RNA next-generation sequencing." _Sci Adv. 2020 Dec 9;6(50):eabb1654_.
    9. Tong et al. "Impact of RNA-seq data analysis algorithms on gene expression estimation and downstream prediction." _Sci Rep. 2020 Oct 21;10(1):17925_.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
