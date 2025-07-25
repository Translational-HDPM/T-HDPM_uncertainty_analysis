import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Modeling of Measurement Uncertainty of a high-dimensional RNA-Seq classifier of cell-free mRNA for Alzheimer’s Disease


    ### Author: Deb Debnath
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Motivation

    The main reasons for ‘clinical-grade’ measurement uncertainty
    usefulness include:

    - improved understanding of test interpretation;
    - operational tool to discern laboratory test drift;
    - sheds light on the analytes that should be prioritized to decrease overall uncertainty range and;
    - upon request, laboratories must make estimates of measurement uncertainty available to laboratory users.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Goal

    Using modeled simulation of high dimensional RNA-Seq, can we estimate inherent empirically informed measurement uncertainties in an illustrative AD classifier?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Measurement Uncertainty: Regulatory Setting

    - Clinical Laboratory Standards Institute (CLSI)
        - CLSI EP29-A Expression of Measurement Uncertainty in Laboratory Medicine
    - International Standards Organization (ISO)
        - ISO 15189-2012
        - ISO/TS 20914:2019
    - Food and Drug Administration (FDA)
        - Class II Special Controls Guidance Document: Ovarian Adnexal Mass Assessment Score Test System (2011).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Monte Carlo techniques are recommended by FDA to estimate diagnostic uncertainty of multi-dimensional classifiers. Overall uncertainty of high dimensional classifiers can be determined or estimated. Besides noted variation, sample site, operator and instrument variation need to be considered."""
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
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The machine learning model, a logistic regression with L2 regularization, was trained using the expression levels (transcripts per million) of these 1658 selected genes from the University of Kentucky training cohort (24 NCI, 66 AD). L2 regularization was specifically employed to prevent overfitting. Metaparameters for this model were optimized using a 15-fold cross-validation strategy on the training cohort. 

    The classifier's ability to discriminate between AD patients and NCI controls was then rigorously evaluated on an independent test set. This test set consisted of the remaining 60 AD patients and 92 NCI controls, derived from four independent sources distinct from the training data (UC San Diego, University of Washington, Indiana University, BioIVT). In this independent validation, the classifier achieved an Area Under the Receiver Operating Characteristic curve (AUC) of 0.83. Further analysis revealed that the genes included in the classifier were enriched in biological pathways known to be associated with AD pathogenesis, including immune response and cellular metabolic processes, thereby lending biological plausibility to the statistical findings
    """
    )
    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    return Path, os


@app.cell
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
        generate_waterfall_plot,
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
        generate_waterfall_plot,
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
    Here we set some parameters that we will use throughout the analysis.

    1. `mean_TPM`: When we filter genes for analysis with a reduced feature set, we drop genes for which the mean TPM is below this cutoff
    2. `num_patients`: Total number of samples/subjects/patients
    3. `uncertainties`: List of values representing maximum percentage of noise/uncertainty to simulate. (aka coefficient of variation). The corresponding slider below can be used to specify the upper and lower limits of percentages, with values being generated in steps of 5.
    4. `n_samples`: Number of Monte Carlo samples to simulate for each subject
    """
    )
    return


@app.cell
def _(os):
    master_seed = 123  # Random number seed
    num_parallel_workers = (
        os.cpu_count()
        # Number of parallel jobs to run for simulation
    )
    return master_seed, num_parallel_workers


@app.cell
def _(
    mean_tpm_slider,
    n_samples_slider,
    num_patients_slider,
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
    num_patients = (
        num_patients_slider.value
    )  # Total number of samples/subjects/patients
    return mean_TPM, n_samples, num_patients, uncertainties


@app.cell
def _(mo, patients_df_1):
    mean_tpm_slider = mo.ui.slider(
        start=0, stop=100, value=0, label="Mean TPM cutoff value", show_value=True
    )
    num_patients_slider = mo.ui.slider(
        start=1,
        stop=patients_df_1.shape[1],
        value=patients_df_1.shape[1],
        label="Number of patients to consider from the dataset",
        show_value=True,
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
        stop=100,
        step=1,
        value=[5, 35],
        label="Range of percent uncertainty values to simulate",
        show_value=True,
    )
    return (
        mean_tpm_slider,
        n_samples_slider,
        num_patients_slider,
        uncertainty_range_slider,
    )


@app.cell(hide_code=True)
def _(
    mean_tpm_slider,
    mo,
    n_samples_slider,
    num_patients_slider,
    uncertainty_range_slider,
):
    mo.callout(
        mo.vstack(
            [
                num_patients_slider,
                mean_tpm_slider,
                n_samples_slider,
                uncertainty_range_slider,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### The Dataset""")
    return


@app.cell(hide_code=True)
def _(mo, patients_df_2):
    mo.md(
        rf"""
    The `raw_data` dataframe contains the TPM values for {patients_df_2.shape[1]} subjects (including technical replicates), along with the coefficients of the classifier. The `pathos` dataframe contains the original categories that the patients belong to.


    For convenience we set the indexes of the dataframes to the patient IDs. We also ensure that between `pathos` and `raw_data` there are no null values and no missing patients (subjects).
    """
    )
    return


@app.cell
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


@app.cell
def _(raw_data):
    raw_data.head()
    return


@app.cell
def _(pathos):
    pathos.head()
    return


@app.cell
def _(pathos):
    pathos_1 = pathos.dropna()
    pathos_1 = pathos_1.loc[pathos_1.index.dropna(), :]
    pathos_1.index = pathos_1.index.astype(int).astype(str)
    return (pathos_1,)


@app.cell
def _(np, raw_data):
    patients_df = raw_data[~raw_data.loc[:, "Coeff"].isnull()]
    coefficients = np.nan_to_num(np.array(patients_df.loc[:, "Coeff"]))
    patients_df = patients_df.filter(regex="^\\d+")
    genes = patients_df.index.values
    return coefficients, genes, patients_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The raw dataset contains technical replicates for some subjects. We average the technical replicates to reduce them to a single data point."""
    )
    return


@app.cell
def _(genes, pathos_1, patients_df):
    grouped_cols = patients_df.columns.str.split("-").str[0]
    grouped = patients_df.groupby(grouped_cols, axis=1)
    patients_df_1 = grouped.apply(lambda x: x.mean(axis=1)).reset_index(drop=True)
    patients_df_1.index = genes

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
    mo.md(r"""### Dropping genes below TPM threshold""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""When analysing a dataset with a reduced set of genes, we keep the genes where the mean TPM value is above the specified cutoff. If the `mean_TPM` value is set to zero, no genes are filtered and the entire dataset is used."""
    )
    return


@app.cell
def _(patients_df_1):
    means = patients_df_1.mean(axis=1)
    return (means,)


@app.cell
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
    For deciding on the probability threshold, we use **Youden's J statistic (Youden's index)** which is defined as

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

    We would like to maximize the Youden's index with our threshold selection. We do this _by selecting a threshold at which sensitivity and specificity are equal_.
    """
    )
    return


@app.cell
def _(antilogit_classifier_score, coefficients_1, np, patients_df_2, pd):
    _z_scores = (
        patients_df_2.values - patients_df_2.mean(axis=1).values.reshape(-1, 1)
    ) / patients_df_2.std(axis=1).values.reshape(-1, 1)
    gt_probs = antilogit_classifier_score(
        np.sum(coefficients_1[:, np.newaxis] * _z_scores, axis=0)
    )
    gt_probs = pd.Series(index=patients_df_2.columns, data=gt_probs)
    return (gt_probs,)


@app.cell
def _(build_sensitivity_specificity_df, gt_probs, pathos_2):
    ad_sens_spec_df = build_sensitivity_specificity_df(pathos_2, gt_probs, "AD")
    nci_sens_spec_df = build_sensitivity_specificity_df(pathos_2, gt_probs, "NCI")
    return ad_sens_spec_df, nci_sens_spec_df


@app.cell
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
    _fig.text(0.5, 0.05, "Probability threshold", ha="center", va="center")
    _fig.text(0.08, 0.35, "Sensitivity/Specificity", rotation="vertical")
    _fig.suptitle(
        "Sensitivity and specificity versus probability threshold curves for AD and NCI categories."
    )
    return


@app.cell
def _(ad_sens_spec_df, np, using_dummy_data):
    _precision = 0 if using_dummy_data else 2
    _rounded_sens = ad_sens_spec_df["sensitivity"].apply(
        lambda x: np.round(x, _precision)
    )
    _rounded_spec = ad_sens_spec_df["specificity"].apply(
        lambda x: np.round(x, _precision)
    )
    _filt = _rounded_sens == _rounded_spec
    assert _filt.sum() >= 1

    target_ad_specificity = (
        ad_sens_spec_df.loc[_filt, "specificity"].values[0] * 100
    )  # For single threshold, roughly where sensitivity = specificity
    single_thres = ad_sens_spec_df.loc[_filt, "threshold"].values[0] / 100
    # print(f"{single_thres=}")
    return single_thres, target_ad_specificity


@app.cell
def _(
    calculate_sensitivity_specificity_and_predictive_values,
    gt_probs,
    mo,
    pathos_2,
    single_thres,
):
    ad_sens, ad_spec, _, _ = calculate_sensitivity_specificity_and_predictive_values(
        pathos_2["Disease"].apply(lambda x: 1 if x == "AD" else 0),
        gt_probs.apply(lambda x: 1 if x >= single_thres else 0),
        0,
    )
    ad_spec = ad_spec * 100.0
    ad_sens = ad_sens * 100.0
    mo.callout(
        mo.md(
            f"At threshold {single_thres:.4f}, AD sensitivity = {ad_sens:.4f}, specificity = {ad_spec:.4f}, Youden's index = {(ad_sens + ad_spec - 100) / 100:.4f}. So, we use {single_thres:.4f} as our probability threshold for the single threshold classification scenario."
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, single_thres):
    mo.md(
        rf"""For two thresholds, we find lower and upper thresholds that maximize Youden's index for NCI and AD classes respectively. However, we are limited by our dataset, since the diagnoses are dichotomised. If there were a third intermediate category between AD and NCI, we could have calculated distinct lower and upper thresholds from the sensitivity and specificity information. However, if we try to find lower and upper thresholds following the criteria stated before, we end up with the same lower and upper threshold. To mitigate this we manually set the lower threshold and upper thresholds farther apart from each other, but in the vicinity of the calculated value of the dichotomous decision threshold ({single_thres:.4f})."""
    )
    return


@app.cell
def _(mo):
    threshold_step_slider = mo.ui.slider(
        start=0.01,
        stop=1.00,
        step=0.005,
        value=0.06,
        show_value=True,
        label="Step to calculate lower and upper thresholds from the dichotomous threshold.",
    )
    threshold_step_slider
    return (threshold_step_slider,)


@app.cell
def _(single_thres, threshold_step_slider):
    dual_thres_low = max(0.01, single_thres - threshold_step_slider.value)
    dual_thres_high = min(1.0, single_thres + threshold_step_slider.value)
    return dual_thres_high, dual_thres_low


@app.cell
def _(
    calculate_sensitivity_specificity_and_predictive_values,
    dual_thres_high,
    dual_thres_low,
    gt_probs,
    mo,
    pathos_2,
):
    nci_sens_low, nci_spec_low, _, _ = (
        calculate_sensitivity_specificity_and_predictive_values(
            pathos_2["Disease"].apply(lambda x: 1 if x == "AD" else 0),
            gt_probs.apply(lambda x: 1 if x >= dual_thres_low else 0),
            0,
        )
    )
    ad_sens_high, ad_spec_high, _, _ = (
        calculate_sensitivity_specificity_and_predictive_values(
            pathos_2["Disease"].apply(lambda x: 1 if x == "AD" else 0),
            gt_probs.apply(lambda x: 1 if x >= dual_thres_high else 0),
            1,
        )
    )
    ad_spec_high = ad_spec_high * 100.0
    ad_sens_high = ad_sens_high * 100.0
    nci_spec_low = nci_spec_low * 100.0
    nci_sens_low = nci_sens_low * 100.0

    mo.callout(
        mo.md(
            f"""
            At lower threshold {dual_thres_low:.4f}, NCI sensitivity = {nci_sens_low:.4f}, specificity = {nci_spec_low:.4f}, Youden's index = {(nci_sens_low + nci_spec_low - 100) / 100:.4f}.\n
            At upper threshold {dual_thres_high:.4f}, AD sensitivity = {ad_sens_high:.4f}, specificity = {ad_spec_high:.4f}, Youden's index = {(ad_sens_high + ad_spec_high - 100) / 100:.4f}
            """
        )
    )
    return ad_spec_high, nci_spec_low


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Performing Monte Carlo simulations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We follow established guidance by the FDA ([Ovarian Adnexal Mass Assessment Score system (2011)](https://www.fda.gov/medical-devices/guidance-documents-medical-devices-and-radiation-emitting-products/ovarian-adnexal-mass-assessment-score-test-system-class-ii-special-controls-guidance-industry-and)) in simulating technical variation in the TPM values."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""In general, given a measurement $\mu$, to simulate $k$ % uncertainty ($k$% coefficient of variation/relative standard deviation) we sample from a Gaussian distribution with mean $\mu$ and standard deviation (SD) $kX/100$."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""However, for RNA-seq datasets, modeling uncertainty in this fashion with a constant noise level ignores the trend of technical variation commonly observed (e.g. in Fig. 1.(a) from [Law et al (2014)](https://link.springer.com/content/pdf/10.1186/gb-2014-15-2-r29.pdf))."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Aligning with Law et al (2014), the relationship between the standard deviation and the mean can be modeled as

    $$
    \sqrt{\sigma} = \frac{a}{b + \mu} + c
    $$

    where $\mu$ and $\sigma$ are mean and standard deviation of the $log_2 (1+TPM)$ dataset, respectively and $a$, $b$, $c$ are constants.\\

    We start with values of $a$ = 0.75, $b$ = 1.0, $c$ = 0.25, $scaling factor$ = 8. 

    ///note
    The values of $a$, $b$, $c$ and $scaling factor$ have been set without empirical calculations due to lack of sufficient technical replicate data. Ideally, if technical replicate data is available, these parameters should be set empirically.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To generate Monte Carlo TPM samples for a given gene $i$ and subject $j$:

    1. Calculate $\sigma_{ij}$ using the SD-mean relationship with $TPM_{ij}$ as the mean:

    $$\mu_{ij} = log_{2}(1 + TPM_{ij})$$

    $$\sigma_{ij} = (\frac{a}{\mu_{ij} + b} + c)^2$$

    $$\sigma_{ij, scaled} = \frac{scaling factor * k * \sigma_{ij}}{100}$$

    2. Generate $N$ samples $x_k$ ($k$ = $1…N$) from a Gaussian distribution with mean = $\mu_{ij}$ and standard deviation = $\sigma_{ij}$, scaled.
    3. Convert samples to TPM scale by exponentiation:
    $$TPM_{k, sampled} = 2^{x_k}$$
    """
    )
    return


@app.cell
def _(np):
    def calculate_scaled_sd(
        tpm: float,
        uncertainty_pct: int | float,
        a: float = 0.75,
        b: float = 1.0,
        c: float = 0.25,
        scaling_factor: float = 8.0,
    ) -> float:
        r"""
        Calculate scaled standard deviation for a given TPM value and baseline
        uncertainty to simulate based on the equation
        $$
        \sqrt{\sigma} = \frac{a}{b + \mu} + c
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
        sqrt_sigma = a / (np.log2(tpm + 1) + b) + c
        scaled_pct_sd = scaling_factor * uncertainty_pct * sqrt_sigma**2.0
        return scaled_pct_sd / 100

    return (calculate_scaled_sd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    After generating the simulated measurements we calculate the classifier predictions for these. We then compare the score for the simulated measurement ("simulated score") against the score for the actual measurement ("actual score"). Based on the set threshold(s) we track the classification of the simulated scores against the actual scores.

    A **differentially classified subject** is a subject (patient) that has one or more simulated scores that produced a different classification into AD/NCI than the actual score. We track differentially classified subjects under two scenarios

    1. at least one of the simulated scores is different from the actual score.
    2. at least 10 % of the simulated scores are different from the actual score.

    The two different definitions of differential classification track the performance of the classifier under a conservative and a relaxed definition of differential classification.
    """
    )
    return


@app.cell
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


@app.cell
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
    res_1_diff_cls = simulate_multiple_uncertainties(
        patients_df_2,
        sampler,
        uncertainties,
        thres_low=dual_thres_low,
        thres_high=dual_thres_high,
        single_thres=single_thres,
        coefficients=coefficients_1,
        diff_class_lim=1,
        n_samples=n_samples,
        seed=master_seed,
        num_workers=num_parallel_workers,
    )
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
    return res, res_1_diff_cls


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualization and post-processing""")
    return


@app.cell
def _(
    plot_differential_classification_results,
    res,
    res_1_diff_cls,
    target_ad_specificity,
    uncertainties,
):
    plot_differential_classification_results(
        gt_labels=res.single_thres_gt_labels,
        one_sim_mismatch_pred_labels_dict=res_1_diff_cls.single_thres_pred_labels,
        ten_pct_sim_mismatch_pred_labels_dict=res.single_thres_pred_labels,
        labels_dict=dict(zip(["NCI", "AD"], "bg")),
        figure_title=f"Differential classification for levels of uncertainty ({min(uncertainties)}-{max(uncertainties)}%)\n"
        + f" (single threshold, specificity: {target_ad_specificity:.2f} % AD)",
    )
    return


@app.cell
def _(
    ad_spec_high,
    nci_spec_low,
    plot_differential_classification_results,
    res,
    res_1_diff_cls,
    uncertainties,
):
    plot_differential_classification_results(
        gt_labels=res.dual_thres_gt_labels,
        one_sim_mismatch_pred_labels_dict=res_1_diff_cls.dual_thres_pred_labels,
        ten_pct_sim_mismatch_pred_labels_dict=res.dual_thres_pred_labels,
        labels_dict=dict(zip(["NCI", "Intermediate", "AD"], "bgm")),
        figure_title=f"Differential classification for levels of uncertainty ({min(uncertainties)}-{max(uncertainties)} %)\n"
        + f" (dual threshold, specificities: {nci_spec_low:.2f}% NCI and {ad_spec_high:.2f}% AD)",
    )
    return


@app.cell
def _(
    ad_spec_high,
    nci_spec_low,
    plot_jaccard_index_plot,
    res,
    target_ad_specificity,
    uncertainties,
):
    plot_jaccard_index_plot(
        labels_dict_single_thres={"NCI": "b", "AD": "g"},
        labels_dict_dual_thres={"NCI": "b", "Intermediate": "r", "AD": "g"},
        gt_labels_single_thres=res.single_thres_gt_labels,
        gt_labels_dual_thres=res.dual_thres_gt_labels,
        pred_labels_dict_single_thres=res.single_thres_pred_labels,
        pred_labels_dict_dual_thres=res.dual_thres_pred_labels,
        single_thres_plot_title=f"Single threshold, specificity: {target_ad_specificity:.2f} % AD",
        dual_thres_plot_title=f"Dual threshold, specificities: {nci_spec_low:.2f}% NCI and {ad_spec_high:.2f}% AD",
        figure_title="Jaccard index plot showing differential classification\n"
        + f"for levels of uncertainty ({min(uncertainties)}-{max(uncertainties)} %)"
        + "(at least 10% simulated scores mismatch)",
    )
    return


@app.cell
def _(generate_waterfall_plot, gt_probs, res, single_thres, uncertainties):
    uncertainty = max(uncertainties)
    generate_waterfall_plot(
        threshold=single_thres,
        probs=gt_probs,
        color_labels_data=res.single_thres_pred_labels[uncertainty],
        labels={0: "NCI", 1: "AD"},
        colors=["#123456", "#fedabc"],
        title=f"Waterfall plot showing simulated and inferent scores\n at {uncertainty}% simulated uncertainty",
        legend_title="Simulated score\n   predictions",
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


@app.cell
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


@app.cell
def _(
    GridSpec,
    ad_spec_high,
    dual_thres_subj_wise_agreement,
    gt_probs,
    nci_spec_low,
    pathos_2,
    plot_v_plot,
    plt,
    single_thres_subj_wise_agreement,
    sns,
    target_ad_specificity,
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
        f"Single threshold, specificity: {target_ad_specificity:.2f}% AD",
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
        f"Dual threshold, specificities: {nci_spec_low:.2f}% NCI and {ad_spec_high:.2f}% AD",
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
        "V-plot showing agreement between simulated and inferent scores"
        + f" for levels of uncertainty ({min(uncertainties)}-{max(uncertainties)}%)"
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
def _(mo):
    mo.md(r"""### Histogram of Simulated Scores""")
    return


@app.cell(hide_code=True)
def _(gt_probs, plt, res, sns):
    _uncert = 35
    _pred_probs = res.pred_prob_arrs[_uncert]
    sns.histplot(
        _pred_probs,
        color="b",
        alpha=0.3,
        fill=True,
        bins=30,
        stat="density",
        label="Simulated subjects",
    )
    sns.histplot(
        gt_probs.values,
        color="r",
        alpha=0.3,
        fill=True,
        bins=30,
        stat="density",
        label="Real subjects",
    )
    plt.xlabel("Classifier score (probability)")
    plt.ylabel("Density")
    plt.legend()
    plt.title(
        f"Histogram of probability scores from \nsimulated and real subjects at {_uncert}% simulated uncertainty."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Preliminary observations

    - RNA-Seq Measurement Uncertainty impacts differential classification predominantly at the classifier threshold.
    - Filtering out genes with low mean TPMs decreases the percentage of both AD and NCI subjects whose diagnostic classification changes when at least 10% of simulations mismatch. 
    - These low-expression genes are a source of classification instability under uncertainty which need to be taken into account when building diagnostic classifications.
    - For a diagnostic classifier with two thresholds, 
        - An initial lab result might suggest a wait and watch approach.
        - The demonstration of near-certain reclassification under typical measurement noise for this specific patient's result acts like a diagnostic stress test (i.e revealing high instability).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### References

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
