import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Adjusting Standard Deviation based on TPM
    ### Author: Deb Debnath
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    return (Path,)


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.dtypes import NumpyFloat32Array1D
    from src.postprocessing import plot_bland_altman

    return NumpyFloat32Array1D, np, pd, plot_bland_altman, plt


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


@app.cell(hide_code=True)
def _(Path, pd):
    data_root = Path(__file__).parent.parent.parent / "raw_data"
    raw_data = pd.read_excel(
        data_root / "ClusterMarkers_1819ADcohort.congregated_DR.xlsx", sheet_name=1
    )
    pathos = pd.read_excel(
        data_root / "Purdue expanded info AD subject sample added Info Apoe.xlsx"
    )
    pathos = pathos.set_index("Isolate ID")
    raw_data = raw_data.set_index("gene_id")
    return pathos, raw_data


@app.cell(hide_code=True)
def _(pathos):
    pathos_1 = pathos.dropna(subset=["Disease"])
    pathos_1 = pathos_1.loc[pathos_1.index.dropna(), :]
    pathos_1.index = pathos_1.index.astype(int).astype(str)
    return (pathos_1,)


@app.cell(hide_code=True)
def _(raw_data):
    patients_df = raw_data[~raw_data.loc[:, "Coeff"].isnull()]
    patients_df = patients_df.filter(regex="^\\d+")
    genes = patients_df.index.values
    return genes, patients_df


@app.cell(hide_code=True)
def _(genes, pathos_1, patients_df):
    grouped_cols = patients_df.columns.str.split("-").str[0]
    grouped = patients_df.groupby(grouped_cols, axis=1)
    patients_df_1 = grouped.apply(lambda x: x.mean(axis=1)).reset_index(drop=True)
    patients_df_1.index = genes

    _patients_df_cols_not_in_pathos = []
    for _col in patients_df_1.columns:
        if _col not in pathos_1.index:
            _patients_df_cols_not_in_pathos.append(_col)

    pathos_2 = pathos_1
    for _col in pathos_1.index:
        if _col not in patients_df_1.columns:
            pathos_2 = pathos_1.drop(_col)

    for _col in _patients_df_cols_not_in_pathos:
        pathos_2.loc[_col, "Disease"] = "NCI"
    patients_df_1 = patients_df_1.loc[:, pathos_2.index]
    return (patients_df_1,)


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


@app.cell(hide_code=True)
def _():
    master_seed = 321  # Random number seed
    return (master_seed,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Dropping genes below TPM threshold""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""When analysing a dataset with a reduced set of genes, we keep the genes where the mean TPM value is above the specified cutoff. If the `mean_TPM` value is set to zero, no genes are filtered and the entire dataset is used."""
    )
    return


@app.cell(hide_code=True)
def _(patients_df_1):
    means = patients_df_1.mean(axis=1)
    return (means,)


@app.cell(hide_code=True)
def _(mean_TPM, means, num_patients, patients_df_1):
    patients_df_2 = patients_df_1[means >= mean_TPM]
    patients_df_2 = patients_df_2.iloc[:, :num_patients]
    return (patients_df_2,)


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
        r"""## Modeling Technical Variability based on "voom" (variance modeling at the observational level)"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We follow established guidance by the FDA ([Ovarian Adnexal Mass Assessment Score system (2011)](https://www.fda.gov/medical-devices/guidance-documents-medical-devices-and-radiation-emitting-products/ovarian-adnexal-mass-assessment-score-test-system-class-ii-special-controls-guidance-industry-and)) in simulating technical variation in the TPM values.

    In general, given a measurement $\mu$, to simulate $k$ % uncertainty ($k$% coefficient of variation/relative standard deviation) we sample from a Gaussian distribution with mean $\mu$ and standard deviation (SD) $kX/100$.

    However, for RNA-seq datasets, modeling uncertainty in this fashion with a constant noise level ignores the trend of technical variation commonly observed (e.g. in Fig. 1.(a) from [Law et al (2014)](https://link.springer.com/content/pdf/10.1186/gb-2014-15-2-r29.pdf)).

    Aligning with Law et al (2014), the relationship between the standard deviation and the mean can be modeled as

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
    To generate Monte Carlo TPM samples for a given gene $i$ and subject $j$:

    1. Calculate $\sigma_{ij}$ using the SD-mean relationship with $TPM_{ij}$ as the mean:

    $$\mu_{ij} = log_{2}(1 + TPM_{ij})$$

    $$\sigma_{ij} = (\frac{a}{\mu_{ij} + b} + c)$$

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
        *,
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
def _(NumpyFloat32Array1D, calculate_scaled_sd, np):
    def sampler(
        tpm: float,
        baseline_rsd: float,
        n_points: int = 1000,
        *,
        a_val: float,
        b_val: float,
        c_val: float,
        scaling_factor: float,
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
        a_val, b_val, c_val, scaling_factor
            Parameters for the SD calculation function
        seed
            Seed for random number generator. Default is None.

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.float32]]
            1D numpy array of floating point values representing TPM samples.
        """
        rng = np.random.default_rng(seed)
        scaled_sd = calculate_scaled_sd(
            tpm,
            baseline_rsd * 100,
            a=a_val,
            b=b_val,
            c=c_val,
            scaling_factor=scaling_factor,
        )
        return np.pow(2.0, rng.normal(np.log2(tpm + 1), scaled_sd, n_points))

    return (sampler,)


@app.cell(hide_code=True)
def _(master_seed, np, pd, plt, sampler):
    def plot_histogram_gene_samples_adaptive_sd(
        patients_df: pd.DataFrame,
        uncertainties: list[int],
        n_samples: int,
        i: int,
        *,
        a_val: float,
        b_val: float,
        c_val: float,
        scaling_factor: float,
        seed: int = master_seed,
    ) -> plt.Figure:
        gene = patients_df.index[i]
        row = patients_df.loc[gene, :].sort_values().values
        quantiles = np.percentile(row, [1, 25, 50, 75, 99])
        labels = [
            "lower fence",
            "25th percentile",
            "50th percentile",
            "75th percentile",
            "upper fence",
        ]
        fig, axs = plt.subplots(
            nrows=len(uncertainties),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(15, 10),
        )
        leg_handles, leg_labels = (None, None)
        for count, uncertainty in enumerate(uncertainties):
            samples = np.zeros((quantiles.shape[0], n_samples))
            for i, val in enumerate(quantiles):
                samples[i] = sampler(
                    val,
                    uncertainty / 100,
                    n_samples,
                    a_val=a_val,
                    b_val=b_val,
                    c_val=c_val,
                    scaling_factor=scaling_factor,
                    seed=seed,
                )
            plt.subplot(len(uncertainties), 1, count + 1)
            for i in range(samples.shape[0]):
                plt.hist(samples[i, :], bins=30, alpha=0.5, label=labels[i])
            plt.title(f"{uncertainty}% uncertainty, scaled")
            if leg_handles is None and leg_labels is None:
                leg_handles, leg_labels = plt.gca().get_legend_handles_labels()
        fig.legend(
            leg_handles,
            leg_labels,
            loc="upper center",
            ncol=len(leg_labels),
            bbox_to_anchor=(0.5, 0.08),
        )
        fig.text(0.5, 0.08, "Simulated TPM counts", va="center", ha="center")
        fig.text(0.08, 0.5, "Counts", va="center", ha="center", rotation="vertical")
        plt.suptitle(
            f"Effect of adaptive standard deviation on sampled data for gene {gene}",
            fontsize=14,
        )
        return fig

    return (plot_histogram_gene_samples_adaptive_sd,)


@app.cell(hide_code=True)
def _(master_seed, np, pd, plt, sampler):
    def plot_variation_of_sd_and_rsd(
        df: pd.DataFrame,
        gene_idx: int,
        n_samples: int,
        uncertainties: list[int],
        *,
        a_val: float,
        b_val: float,
        c_val: float,
        scaling_factor: float,
        seed: int = master_seed,
    ) -> plt.Figure:
        row = df.iloc[gene_idx]
        fig = plt.figure(figsize=(15, 6))
        for uncert in uncertainties:
            samples = np.zeros((row.shape[0], n_samples))
            for i, val in enumerate(row):
                samples[i] = sampler(
                    val,
                    uncert / 100,
                    n_samples,
                    a_val=a_val,
                    b_val=b_val,
                    c_val=c_val,
                    scaling_factor=scaling_factor,
                    seed=seed,
                )
            plt.subplot(121)
            plt.scatter(samples.mean(axis=1), samples.std(axis=1), label=f"{uncert} %")
            plt.subplot(122)
            plt.scatter(
                samples.mean(axis=1),
                samples.std(axis=1) / samples.mean(axis=1) * 100,
                label=f"{uncert} %",
            )
        plt.subplot(121)
        plt.ylabel("Sample standard deviation")
        plt.subplot(122)
        plt.ylabel("Percent relative standard deviation")
        leg_handles, leg_labels = plt.gca().get_legend_handles_labels()
        fig.legend(
            leg_handles,
            leg_labels,
            loc="upper center",
            ncol=len(leg_labels),
            bbox_to_anchor=(0.5, 0.04),
        )
        fig.suptitle(
            f"Variation of SD and RSD versus mean of samples from gene {df.index[gene_idx]}"
        )
        fig.text(0.5, 0.06, "Sample mean", va="center", ha="center")
        return fig

    return (plot_variation_of_sd_and_rsd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Different values for parameters $a$, $b$ and $c$ can have varying effects on simulated TPM values at different percentile levels for a given gene. Experiment with the values in the interactive app below."""
    )
    return


@app.cell(hide_code=True)
def _(mo, patients_df_2):
    gene_idx_slider = mo.ui.slider(
        start=0,
        stop=patients_df_2.shape[0] - 1,
        step=1,
        value=0,
        label="Gene index",
        show_value=True,
    )
    a_slider = mo.ui.slider(
        start=0.0,
        stop=5.0,
        step=0.01,
        value=0.75,
        label="a",
        show_value=True,
    )
    b_slider = mo.ui.slider(
        start=0.0,
        stop=100.0,
        step=0.01,
        value=1.0,
        label="b",
        show_value=True,
    )
    c_slider = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.01,
        value=0.25,
        label="c",
        show_value=True,
    )
    scaling_factor_slider = mo.ui.slider(
        start=1.0,
        stop=10.0,
        step=0.1,
        value=6.0,
        label="scaling_factor",
        show_value=True,
    )
    return a_slider, b_slider, c_slider, gene_idx_slider, scaling_factor_slider


@app.cell(hide_code=True)
def _(
    a_slider,
    b_slider,
    c_slider,
    gene_idx_slider,
    mo,
    scaling_factor_slider,
):
    mo.callout(
        mo.vstack(
            [gene_idx_slider, a_slider, b_slider, c_slider, scaling_factor_slider]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Effect of a, b, c and scaling_factor on calculated SD value""")
    return


@app.cell(hide_code=True)
def _(
    a_slider,
    b_slider,
    c_slider,
    calculate_scaled_sd,
    np,
    patients_df_2,
    plt,
    scaling_factor_slider,
):
    _uncertainty = 25
    _x = np.linspace(1, patients_df_2.max(), 100)
    _fig = plt.figure(figsize=(12, 6))
    plt.scatter(
        _x,
        calculate_scaled_sd(
            _x,
            _uncertainty,
            a=a_slider.value,
            b=b_slider.value,
            c=c_slider.value,
            scaling_factor=scaling_factor_slider.value,
        ),
    )
    plt.semilogx()
    plt.xlabel("Mean TPM")
    plt.ylabel("$SD$")
    plt.ylim([0, 50])
    plt.title(
        r"Effect of $a$, $b$, $c$ and $scaling\_factor$ on calculated SD"
        + "\n"
        + rf"$a$ = {a_slider.value}, $b$ = {b_slider.value}, $c$ = {c_slider.value}, $scaling\_factor$ = {scaling_factor_slider.value}"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""### Effect of a, b, c and scaling_factor on distribution of Simulated TPM values"""
    )
    return


@app.cell(hide_code=True)
def _(
    a_slider,
    b_slider,
    c_slider,
    gene_idx_slider,
    n_samples,
    patients_df_2,
    plot_histogram_gene_samples_adaptive_sd,
    scaling_factor_slider,
    uncertainties,
):
    plot_histogram_gene_samples_adaptive_sd(
        patients_df_2,
        [
            uncertainties[0],
            uncertainties[len(uncertainties) // 2],
            uncertainties[-1],
        ],
        n_samples,
        gene_idx_slider.value,
        a_val=a_slider.value,
        b_val=b_slider.value,
        c_val=c_slider.value,
        scaling_factor=scaling_factor_slider.value,
    )
    return


@app.cell(hide_code=True)
def _(
    a_slider,
    b_slider,
    c_slider,
    gene_idx_slider,
    n_samples,
    patients_df_2,
    plot_variation_of_sd_and_rsd,
    scaling_factor_slider,
    uncertainties,
):
    plot_variation_of_sd_and_rsd(
        patients_df_2,
        gene_idx_slider.value,
        n_samples,
        [
            uncertainties[0],
            uncertainties[len(uncertainties) // 2],
            uncertainties[-1],
        ],
        a_val=a_slider.value,
        b_val=b_slider.value,
        c_val=c_slider.value,
        scaling_factor=scaling_factor_slider.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""### Do the simulated replicates accurately model the Technical variation in the Dataset?"""
    )
    return


@app.cell(hide_code=True)
def _(mo, patients_df):
    cols_with_replicates = list(
        set([_col.split("-")[0] for _col in patients_df.columns if _col.endswith("r2")])
    )
    patient_id_w_replicates_col_slider = mo.ui.slider(
        start=0,
        stop=len(cols_with_replicates) - 1,
        step=1,
        value=0,
        label="Patient ID index with replicates",
        show_value=True,
    )
    return cols_with_replicates, patient_id_w_replicates_col_slider


@app.cell(hide_code=True)
def _(mo, patient_id_w_replicates_col_slider):
    mo.callout(patient_id_w_replicates_col_slider)
    return


@app.cell(hide_code=True)
def _(
    NumpyFloat32Array1D,
    a_slider,
    b_slider,
    c_slider,
    np,
    pd,
    sampler,
    scaling_factor_slider,
):
    def get_simulated_replicate(
        df: pd.DataFrame,
        patient_id: str,
        uncertainty: float,
        *,
        a_val: float = a_slider.value,
        b_val: float = b_slider.value,
        c_val: float = c_slider.value,
        scaling_factor: float = scaling_factor_slider.value,
    ) -> NumpyFloat32Array1D:
        """
        Helper function to generate a simulated technical replicate for a given subject.
        """
        tpm_vals = df[patient_id]
        ret = np.zeros_like(tpm_vals)
        for i, tpm in enumerate(tpm_vals):
            ret[i] = sampler(
                tpm,
                uncertainty / 100,
                1,
                a_val=a_val,
                b_val=b_val,
                c_val=c_val,
                scaling_factor=scaling_factor,
            )[0]
        return ret

    return (get_simulated_replicate,)


@app.cell(hide_code=True)
def _(NumpyFloat32Array1D, np):
    def get_loa(
        meas_1: NumpyFloat32Array1D, meas_2: NumpyFloat32Array1D
    ) -> tuple[float, float]:
        """Compute Limits of agreement of two measurement arrays"""
        differences = meas_1 - meas_2
        # Compute statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        # Limits of agreement (mean difference ± 1.96*SD)
        return mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff

    return (get_loa,)


@app.cell(hide_code=True)
def _(
    cols_with_replicates,
    get_loa,
    get_simulated_replicate,
    patient_id_w_replicates_col_slider,
    patients_df,
    patients_df_2,
    plot_bland_altman,
    plt,
    uncertainties,
):
    _col = cols_with_replicates[patient_id_w_replicates_col_slider.value]
    _fig, _ = plt.subplots(figsize=(18, 6), nrows=1, ncols=4, sharex=True, sharey=True)
    _uncert_low, _uncert_mid, _uncert_high = (
        uncertainties[0],
        uncertainties[len(uncertainties) // 2],
        uncertainties[-1],
    )
    _y_min, _y_max = float("inf"), -float("inf")
    for _i, _uncert in enumerate([_uncert_low, _uncert_mid, _uncert_high]):
        plt.subplot(1, 4, _i + 1)
        _meas_1, _meas_2 = (
            patients_df_2[_col],
            get_simulated_replicate(patients_df_2, _col, _uncert),
        )
        plot_bland_altman(
            _meas_1,
            _meas_2,
            f"Simulated replicates at {_uncert}% uncertainty",
            save=False,
            show=False,
        )

        _loa_lower, _loa_upper = get_loa(_meas_1, _meas_2)
        _y_min = min(_y_min, _loa_lower)
        _y_max = max(_y_max, _loa_upper)

        plt.xlabel("")
        plt.ylabel("")
    plt.subplot(144)
    plot_bland_altman(
        patients_df[f"{_col}-r1"],
        patients_df[f"{_col}-r2"],
        "Actual technical replicates",
        save=False,
        show=False,
    )
    _loa_lower, _loa_upper = get_loa(_meas_1, _meas_2)
    _y_min = min(_y_min, _loa_lower)
    _y_max = max(_y_max, _loa_upper)

    plt.xlabel("")
    plt.ylabel("")
    plt.ylim([1.5 * _y_min, 1.5 * _y_max])
    _fig.text(0.5, 0.02, "Difference between measurements", va="center", ha="center")
    _fig.text(
        0.1,
        0.5,
        "Mean of measurements",
        va="center",
        ha="center",
        rotation="vertical",
    )
    _fig.suptitle(
        f"Bland-Altman plot showing simulated and actual technical replicates under different uncertainty levels (Patient ID {_col})",
        fontsize=14,
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
