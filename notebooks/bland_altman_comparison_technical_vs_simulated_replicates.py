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
    import os
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

    return NumpyFloat32Array1D, np, pd, plot_bland_altman


@app.cell(hide_code=True)
def _(Path, pd):
    raw_data, pathos = None, None
    data_root = Path(__file__).parent.parent / "data"
    raw_data = pd.read_excel(
        data_root / "raw_data.xlsx",
        sheet_name=1,
    )
    raw_data = raw_data.set_index("gene_id")
    return (raw_data,)


@app.cell(hide_code=True)
def _(np, raw_data):
    patients_df = raw_data[~raw_data.loc[:, "Coeff"].isnull()]
    coefficients = np.nan_to_num(np.array(patients_df.loc[:, "Coeff"]))
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

    def sampler_gaussian(
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
            Reference uncertainty value. Must be between 0 and 1.
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
        return rng.normal(tpm, baseline_rsd * tpm, n_points)

    return (sampler,)


@app.cell
def _(patient_ids_with_replicates, random):
    pct_uncertainty = 25
    master_seed = 123
    patient_idx = random.choice(patient_ids_with_replicates)
    return master_seed, patient_idx, pct_uncertainty


@app.cell
def _(patient_idx, patients_df, plot_bland_altman):
    rep_1, rep_2 = (
        patients_df.loc[:, f"{patient_idx}-r1"].values,
        patients_df.loc[:, f"{patient_idx}-r2"],
    )
    plot_bland_altman(
        rep_1,
        rep_2,
        title=f"Bland-Altman plot for technical replicates for subject {patient_idx}",
    )
    return rep_1, rep_2


@app.cell
def _(
    master_seed,
    np,
    patient_idx,
    pct_uncertainty,
    plot_bland_altman,
    rep_1,
    rep_2,
    sampler,
):
    rng = np.random.default_rng(seed=master_seed)
    seed_1, seed_2 = rng.integers(25536), rng.integers(25536)

    sim_rep_1 = np.hstack(
        [
            sampler(tpm, baseline_rsd=pct_uncertainty / 100, n_points=1, seed=seed_1)
            for tpm in rep_1
        ]
    )
    sim_rep_2 = np.hstack(
        [
            sampler(tpm, baseline_rsd=pct_uncertainty / 100, n_points=1, seed=seed_2)
            for tpm in rep_2
        ]
    )

    plot_bland_altman(
        sim_rep_1,
        sim_rep_2,
        title=f"Bland-Altman plot for simulated replicates for subject {patient_idx}\n at {pct_uncertainty}% uncertainty",
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
