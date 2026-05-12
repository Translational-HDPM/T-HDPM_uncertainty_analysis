import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## Modeling of Measurement Uncertainty of a high-dimensional RNA-Seq classifier of cell-free mRNA for Alzheimer’s Disease"""
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

    from src.dtypes import NumpyFloat32Array1D
    from src.simulation import simulate_multiple_uncertainties

    return NumpyFloat32Array1D, np, pd, plt, simulate_multiple_uncertainties


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
        data_root = Path(__file__).parent.parent / "dummy_data"
        raw_data = pd.read_csv(data_root / "tpm_expression_data.csv")
        pathos = pd.read_csv(data_root / "disease_status_data.csv")
    return pathos, raw_data


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
                lambda x: (
                    x[x.index.str.endswith("r2")]
                    if len(x) > 1
                    else x[x.index.str.endswith("r1")]
                )
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
    mo.md(r"""### Performing Monte Carlo simulations""")
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
def _(mo):
    mo.md(
        r"""### How does the threshold selection affect differential classification for each category?"""
    )
    return


@app.cell(hide_code=True)
def _(
    Path,
    coefficients_1,
    collapse_replicates_by,
    master_seed,
    mean_TPM,
    n_samples,
    np,
    num_parallel_workers,
    patients_df_2,
    pd,
    sampler,
    simulate_multiple_uncertainties,
    uncertainties,
):
    import math

    from src.postprocessing import get_differential_classification

    lower = np.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    upper = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99])
    uncertainties_to_simulate = uncertainties[:]
    name = f"differential_classification_dual_threshold_n_{n_samples}_mean_tpm_{mean_TPM}_collapse_replicates_by_{collapse_replicates_by}.csv"
    save_root_dir = Path(__file__).parent.parent / "generated_data"
    save_root_dir.mkdir(exist_ok=True)
    csv_save_path = save_root_dir / name

    if csv_save_path.exists():
        diff_cls_df_full = pd.read_csv(csv_save_path)
    else:
        diff_cls_df_full = pd.DataFrame(
            columns=[
                "lower threshold",
                "upper threshold",
                "uncertainty",
                "NCI",
                "Intermediate",
                "AD",
            ]
        )
        results_dict = dict()
        _idx = 0
        for thres_low in lower:
            for thres_high in upper[upper > thres_low]:
                _res = simulate_multiple_uncertainties(
                    patients_df_2,
                    sampler,
                    uncertainties_to_simulate,
                    thres_low=thres_low,
                    thres_high=thres_high,
                    single_thres=thres_low,
                    coefficients=coefficients_1,
                    diff_class_lim=int(0.1 * n_samples),
                    n_samples=n_samples,
                    seed=master_seed,
                    num_workers=num_parallel_workers,
                )
                results_dict[(thres_low, thres_high)] = _res
                _diff_cls_df = get_differential_classification(
                    _res.dual_thres_gt_labels,
                    _res.dual_thres_pred_labels,
                    ["NCI", "Intermediate", "AD"],
                )
                _num_gt_nci, _num_gt_interm, _num_gt_ad = (
                    (_res.dual_thres_gt_labels == 0).sum(),
                    (_res.dual_thres_gt_labels == 1).sum(),
                    (_res.dual_thres_gt_labels == 2).sum(),
                )
                for _uncertainty, _nci, _interm, _ad in _diff_cls_df.itertuples():
                    _nci, _interm, _ad = (
                        _nci * _num_gt_nci / 100,
                        _interm * _num_gt_interm / 100,
                        _ad * _num_gt_ad / 100,
                    )
                    diff_cls_df_full.loc[_idx] = {
                        "lower threshold": thres_low,
                        "upper threshold": thres_high,
                        "uncertainty": _uncertainty,
                        "NCI": int(_nci if not math.isnan(_nci) else 0),
                        "Intermediate": int(_interm if not math.isnan(_interm) else 0),
                        "AD": int(_ad if not math.isnan(_ad) else 0),
                    }
                    _idx += 1
                diff_cls_df_full.to_csv(csv_save_path)
                print(
                    f"It. {_idx}, low = {thres_low}, high = {thres_high}: Saved CSV to {str(csv_save_path)}."
                )
    return diff_cls_df_full, lower, save_root_dir, upper


@app.cell(hide_code=True)
def _(mo):
    cat_select = mo.ui.dropdown(
        options=["NCI", "Intermediate", "AD", "all"],
        value="all",
        allow_select_none=False,
        label="Select categories to plot: ",
    )
    show_lines_switch = mo.ui.switch(value=False, label="Show lines? ")
    mo.vstack([cat_select, show_lines_switch])
    return cat_select, show_lines_switch


@app.cell(hide_code=True)
def _(
    cat_select,
    collapse_replicates_by,
    diff_cls_df_full,
    mean_TPM,
    n_samples,
    np,
    plt,
    save_root_dir,
    show_lines_switch,
    uncertainties,
    upper,
):
    cats = None
    if cat_select.value in ["NCI", "Intermediate", "AD"]:
        cats = [cat_select.value]
    elif cat_select.value == "all":
        cats = ["NCI", "Intermediate", "AD"]

    _upper_thres = np.max(upper)
    _cmap = {"NCI": "blue", "Intermediate": "yellow", "AD": "red"}

    _fig = plt.figure(figsize=(10, 6))
    for _cat in cats:
        _plot_data = []
        for _uncert in uncertainties:
            _to_plot_df = (
                diff_cls_df_full.groupby(by=["uncertainty", "upper threshold"])
                .get_group((_uncert, _upper_thres))[
                    ["lower threshold", "NCI", "Intermediate", "AD"]
                ]
                .set_index("lower threshold")
            )
            _plot_data.append(_to_plot_df[_cat])
            if show_lines_switch.value:
                plt.plot(
                    _to_plot_df.index,
                    _to_plot_df[_cat],
                    label=_cat,
                    color=_cmap[_cat],
                    alpha=0.1,
                )
        _data = np.vstack(_plot_data)
        _mins = np.min(_data, axis=0)
        _maxs = np.max(_data, axis=0)
        plt.fill_between(
            _to_plot_df.index,
            _mins,
            _maxs,
            facecolor=_cmap[_cat],
            alpha=0.5,
            label=_cat,
        )
    plt.ylim([0, 100])
    plt.ylabel("Number of subjects in category differentially classified")
    plt.title(
        f"Effect of changing lower threshold on differential classification, \nupper threshold = {_upper_thres}, uncertainties {min(uncertainties)} -{max(uncertainties)} %"
    )
    _leg_handles, _leg_labels = plt.gca().get_legend_handles_labels()
    _fig.legend(
        _leg_handles
        if not show_lines_switch.value
        else _leg_handles[len(uncertainties) :: len(uncertainties) + 1],
        _leg_labels
        if not show_lines_switch.value
        else _leg_labels[len(uncertainties) :: len(uncertainties) + 1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
    )

    _fig.savefig(
        save_root_dir
        / f"Effect_of_lower_thres_differential_classification_dual_threshold_n_{n_samples}_mean_tpm_{mean_TPM}_collapse_replicates_by_{collapse_replicates_by}.png"
    )
    _fig
    return (cats,)


@app.cell(hide_code=True)
def _(
    cats,
    collapse_replicates_by,
    diff_cls_df_full,
    lower,
    mean_TPM,
    n_samples,
    np,
    plt,
    save_root_dir,
    show_lines_switch,
    uncertainties,
):
    _lower_thres = np.min(lower)
    _cmap = {"NCI": "blue", "Intermediate": "yellow", "AD": "red"}
    plt.figure(figsize=(10, 6))
    _fig = plt.figure(figsize=(10, 6))
    for _cat in cats:
        _plot_data = []
        for _uncert in uncertainties:
            _to_plot_df = (
                diff_cls_df_full.groupby(by=["uncertainty", "lower threshold"])
                .get_group((_uncert, _lower_thres))[
                    ["upper threshold", "NCI", "Intermediate", "AD"]
                ]
                .set_index("upper threshold")
            )
            _plot_data.append(_to_plot_df[_cat])
            if show_lines_switch.value:
                plt.plot(
                    _to_plot_df.index,
                    _to_plot_df[_cat],
                    label=_cat,
                    color=_cmap[_cat],
                    alpha=0.1,
                )
            _data = np.vstack(_plot_data)
            _mins = np.min(_data, axis=0)
            _maxs = np.max(_data, axis=0)
        plt.fill_between(
            _to_plot_df.index,
            _mins,
            _maxs,
            facecolor=_cmap[_cat],
            alpha=0.5,
            label=_cat,
        )

    plt.ylim([0, 100])
    plt.ylabel("Number of subjects in category differentially classified")
    plt.title(
        f"Effect of changing upper threshold on differential classification, \nlower threshold = {_lower_thres}, uncertainties {min(uncertainties)} -{max(uncertainties)} %"
    )
    _leg_handles, _leg_labels = plt.gca().get_legend_handles_labels()
    _fig.legend(
        _leg_handles
        if not show_lines_switch.value
        else _leg_handles[len(uncertainties) :: len(uncertainties) + 1],
        _leg_labels
        if not show_lines_switch.value
        else _leg_labels[len(uncertainties) :: len(uncertainties) + 1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
    )
    _fig.savefig(
        save_root_dir
        / f"Effect_of_upper_thres_differential_classification_dual_threshold_n_{n_samples}_mean_tpm_{mean_TPM}_collapse_replicates_by_{collapse_replicates_by}.png"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(coefficients_1, np, patients_df_2, pd):
    from src.logreg_classifier import antilogit_classifier_score

    _z_scores = (
        patients_df_2.values - patients_df_2.mean(axis=1).values.reshape(-1, 1)
    ) / patients_df_2.std(axis=1).values.reshape(-1, 1)
    gt_probs = antilogit_classifier_score(
        np.sum(coefficients_1[:, np.newaxis] * _z_scores, axis=0)
    )
    gt_probs = pd.Series(index=patients_df_2.columns, data=gt_probs)
    return (gt_probs,)


@app.cell(hide_code=True)
def _(gt_probs, np, plt, save_root_dir):
    _sorted_probs = gt_probs.sort_values()
    _lower_thres, _upper_thres = 0.10, 0.90

    def _select_label(
        prob: float,
        lower_thres: float = _lower_thres,
        upper_thres: float = _upper_thres,
    ) -> str:
        if prob < lower_thres:
            return "NCI"
        elif prob < upper_thres:
            return "Intermediate"
        else:
            return "AD"

    _labels = _sorted_probs.map(_select_label)
    _fig = plt.figure(figsize=(10, 6))
    _acc = 0
    for _label, _color in zip(["NCI", "Intermediate", "AD"], "bkr"):
        _subset = _labels == _label
        _y = _sorted_probs.loc[_subset]
        plt.scatter(
            np.arange(_acc, _acc + len(_y)),
            _y,
            c=_color,
            label=_label,
        )
        _acc += len(_y)
    plt.xlabel("Patient")
    plt.ylabel("Classifier score (probability)")
    plt.legend()
    plt.title(
        f"Classifier scores for dual threshold classification\nat lower threshold {_lower_thres} and upper threshold {_upper_thres}"
    )
    _fig.text(
        0.25,
        -0.05,
        "Classifier predictions: "
        + "".join(
            [
                f"{_label} = {(_labels == _label).sum()} "
                for _label in ["NCI", "Intermediate", "AD"]
            ]
        ),
        bbox=dict(facecolor="none", edgecolor="k"),
    )
    _fig.savefig(save_root_dir / "classifier_scores.png")
    _fig
    return


@app.cell(hide_code=True)
def _(
    collapse_replicates_by,
    gt_probs,
    mean_TPM,
    n_samples,
    pathos_2,
    plt,
    save_root_dir,
):
    import seaborn as sns

    ad_probs = gt_probs[pathos_2[pathos_2["Disease"] == "AD"].index]
    nci_probs = gt_probs[pathos_2[pathos_2["Disease"] == "NCI"].index]

    plt.figure(figsize=(10, 8))
    sns.histplot(
        ad_probs,
        color="r",
        bins=30,
        label="AD",
        fill=True,
        alpha=0.3,
    )
    sns.histplot(
        nci_probs,
        color="b",
        bins=30,
        label="NCI",
        fill=True,
        alpha=0.3,
    )
    plt.xlim([0.0, 1.0])
    plt.xlabel("Classifier score (probability)")
    plt.ylabel("Number of subjects")
    plt.legend()
    plt.savefig(
        save_root_dir
        / f"histogram of classifier scores_n_{n_samples}_mean_tpm_cutoff_{mean_TPM}_collapse_replicates_by_{collapse_replicates_by}.png"
    )
    plt.show()
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
