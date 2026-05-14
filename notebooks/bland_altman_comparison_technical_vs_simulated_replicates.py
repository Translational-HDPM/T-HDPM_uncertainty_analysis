import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison of Simulated Replicates against Actual Technical Replicates
    """)
    return


@app.cell(hide_code=True)
def _():
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    return Path, random


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.dtypes import NumpyFloat32Array1D
    from src.postprocessing import plot_bland_altman

    return NumpyFloat32Array1D, np, pd, plot_bland_altman, plt


@app.cell(hide_code=True)
def _(Path, pd):
    raw_data = None
    data_root = Path(__file__).parent.parent / "data"
    raw_data = pd.read_excel(
        data_root / "raw_data.xlsx",
        sheet_name=1,
    )
    raw_data = raw_data.set_index("gene_id")
    return (raw_data,)


@app.cell(hide_code=True)
def _(raw_data):
    patients_df = raw_data[~raw_data.loc[:, "Coeff"].isnull()]
    patients_df = patients_df.filter(regex="^\\d+")
    return (patients_df,)


@app.cell(hide_code=True)
def _(patients_df):
    patients_df.head()
    return


@app.cell(hide_code=True)
def _(patients_df):
    patient_ids_with_replicates = [
        col.split("-")[0] for col in patients_df.columns if col.endswith("r2")
    ]
    return (patient_ids_with_replicates,)


@app.cell(hide_code=True)
def _(NumpyFloat32Array1D, np):
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


    def sampler(
        rng: np.random.Generator,
        tpm: float,
        baseline_rsd: float,
        n_points: int = 1000,
    ) -> NumpyFloat32Array1D:
        """
        Function to generate Monte Carlo TPM samples given a TPM value and a
        baseline uncertainty value.

        Parameters
        ----------
        rng
            Random number generator.
        tpm
            TPM value to generate simulated TPM values from.
        baseline_rsd
            Reference uncertainty value to calculate the scaled SD for the simulated
            TPM values. Must be between 0 and 1.
        n_points
            Number of simulated TPM values to generate. Defaults to 1000.

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.float32]]
            1D numpy array of floating point values representing TPM samples.
        """
        scaled_sd = calculate_scaled_sd(tpm, baseline_rsd * 100)
        return np.pow(2.0, rng.normal(np.log2(tpm + 1), scaled_sd, n_points))

    return (sampler,)


@app.cell
def _(np, patient_ids_with_replicates, random):
    uncertainties = [5, 20, 35]
    master_seed = 123
    patient_idx = random.choice(patient_ids_with_replicates)
    rng = np.random.default_rng(seed=master_seed)
    return patient_idx, rng, uncertainties


@app.cell
def _(patient_idx, patients_df):
    rep_1, rep_2 = (
        patients_df.loc[:, f"{patient_idx}-r1"].values,
        patients_df.loc[:, f"{patient_idx}-r2"],
    )
    return rep_1, rep_2


@app.cell(hide_code=True)
def _(
    np,
    patient_idx,
    plot_bland_altman,
    plt,
    rep_1,
    rep_2,
    rng,
    sampler,
    uncertainties,
):
    seed_1, seed_2 = rng.integers(25536, size=2)

    fig, axs = plt.subplots(
        1, len(uncertainties) + 1, sharex=True, sharey=True, figsize=(15, 8)
    )
    y_min, y_max = float("inf"), float("-inf")

    plt.sca(axs[0])
    plot_bland_altman(rep_1, rep_2, title="Technical replicates", show=False)
    plt.xlabel("")
    plt.ylabel("")
    y_min_curr, y_max_curr = plt.ylim()
    y_min = min(y_min, y_min_curr)
    y_max = max(y_max, y_max_curr)
    plt.legend(loc="upper right")

    for i, pct_uncertainty in enumerate(uncertainties):
        rng_1 = np.random.default_rng(seed_1)
        rng_2 = np.random.default_rng(seed_2)
        sim_rep_1 = np.hstack(
            [
                sampler(rng_1, tpm, baseline_rsd=pct_uncertainty / 100, n_points=1)
                for tpm in rep_1
            ]
        )
        sim_rep_2 = np.hstack(
            [
                sampler(rng_2, tpm, baseline_rsd=pct_uncertainty / 100, n_points=1)
                for tpm in rep_1
            ]
        )

        plt.sca(axs[i + 1])
        plot_bland_altman(
            sim_rep_1,
            sim_rep_2,
            title=f"{pct_uncertainty}% uncertainty",
            show=False,
        )
        plt.xlabel("")
        plt.ylabel("")
        y_min_curr, y_max_curr = plt.ylim()
        y_min = min(y_min, y_min_curr)
        y_max = max(y_max, y_max_curr)
        plt.legend(loc="upper right")

    plt.ylim([1.5 * y_min, 1.5 * y_max])
    fig.supxlabel("Average of measurements")
    fig.supylabel("Difference between measurements")
    fig.suptitle(
        f"Bland-Altman plots for technical and synthetic (simulated) replicates for subject {patient_idx}"
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
