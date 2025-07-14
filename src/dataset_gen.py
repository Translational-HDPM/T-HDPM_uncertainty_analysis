"""
Generates a synthetic dataset for demonstration in absence of the real dataset.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def generate_tpm_data(
    num_genes: int = 950, num_patients: int = 250, num_replicates: int = 50
) -> pd.DataFrame:
    """
    Generates a simulated TPM dataset for an RNA-seq experiment with values
    ranging from 0.0 to 900.0.

    Parameters
    ----------
    num_genes
        The number of genes in the dataset.

    num_patients
        The total number of patients.

    num_replicates
        The number of patients with technical replicates.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with TPM values, where rows are genes
        and columns are patient samples.
    """
    genes = [f"Gene {i + 1}" for i in range(num_genes)]
    patient_ids = [f"{i + 1}" for i in range(num_patients)]

    columns = []

    # Add patients with two replicates first
    for i in range(num_replicates):
        columns.append(f"{patient_ids[i]}-r1")
        columns.append(f"{patient_ids[i]}-r2")

    # Add patients with single samples
    for i in range(num_replicates, num_patients):
        columns.append(f"{patient_ids[i]}-r1")

    tpm_df = pd.DataFrame(index=genes, columns=columns)

    # Generate and Populate TPM values
    for gene in genes:
        distribution_choice = np.random.choice(["neg_binomial", "poisson"])

        if distribution_choice == "neg_binomial":
            # Parameters for Negative Binomial are adjusted to keep values
            # generally within the desired range.
            n = np.random.randint(50, 200)
            p = np.random.uniform(0.2, 0.5)
            expression_values = np.random.negative_binomial(n, p, size=len(columns))
        else:
            # Parameter for Poisson is set
            # to a random value to create variety.
            lam = np.random.uniform(200, 400)
            expression_values = np.random.poisson(lam, size=len(columns))

        # Clip values to ensure they are within the 0.0 to 900.0 range
        tpm_df.loc[gene] = np.clip(expression_values, 0, 900).astype(float)
    tpm_df.index.name = "gene_id"
    return tpm_df


def generate_disease_status(num_patients: int = 250) -> pd.DataFrame:
    """
    Generates a dataset predicting disease status for each patient.

    Parameters
    ----------
    num_patients
        The total number of patients.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with patient IDs as the index and
        a single column for disease status.
    """
    patient_ids = [f"{i + 1}" for i in range(num_patients)]

    # Randomly assign "Diseased" or "Not diseased"
    status = np.random.choice(["AD", "NCI"], size=num_patients, p=[0.45, 0.55])

    disease_df = pd.DataFrame({"Disease": status}, index=patient_ids)
    disease_df.index.name = "Isolate ID"

    return disease_df


def train_classifier(
    tpm_expression_data: pd.DataFrame, disease_status_data: pd.DataFrame
) -> pd.Series:
    """
    Trains a logistic regression classifier with L2 regularization and
    15 fold cross validation following Toden et al (2020).

    Parameters
    ----------
    tpm_expression_data
        Dataframe containing TPM values.

    disease_status_data
        Dataframe containing disease status of each patient.

    Returns
    -------
    pd.Series
        A pandas Series containing coefficients for the classifier trained.
    """
    clf = LogisticRegressionCV(cv=15, random_state=0, fit_intercept=False)
    patient_ids_with_replicates = [
        col.split("-")[0] for col in tpm_expression_data.columns if col.endswith("-r2")
    ]
    cols = {}
    for col in tpm_expression_data.columns:
        patient_id = col.split("-")[0]
        if patient_id in patient_ids_with_replicates:
            cols[patient_id] = (
                tpm_expression_data[f"{patient_id}-r1"]
                + tpm_expression_data[f"{patient_id}-r2"]
            ) / 2.0
        else:
            cols[patient_id] = tpm_expression_data[col].copy()
    x = pd.DataFrame(cols)
    y = disease_status_data["Disease"].apply(lambda x: 1 if x == "AD" else 0)

    x_scaled = StandardScaler().fit_transform(x.values.T)
    clf.fit(x_scaled, y.values)
    return pd.Series(index=tpm_expression_data.index, data=clf.coef_.flatten())


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # Generate both datasets
    tpm_expression_data = generate_tpm_data()
    disease_status_data = generate_disease_status()

    # Train a logistic regression classifier on the data with
    # 15 fold cross validation
    coeff = train_classifier(tpm_expression_data, disease_status_data)

    # Add coefficients to TPM dataset
    tpm_expression_data["Coeff"] = coeff

    print("TPM Expression Data")
    print("Shape:", tpm_expression_data.shape)
    print("Head:\n", tpm_expression_data.head())

    tpm_expression_data.to_csv(f"{root_dir}/tpm_expression_data.csv")
    print(f"\nSuccessfully saved TPM data to '{root_dir}/tpm_expression_data.csv'")

    print("\n" + "=" * 40 + "\n")

    print("Disease Status Data")
    print("Shape:", disease_status_data.shape)
    print("Head:\n", disease_status_data.head())

    disease_status_data.to_csv(f"{root_dir}/disease_status_data.csv")
    print(
        f"\nSuccessfully saved disease status data to '{root_dir}/disease_status_data.csv'"
    )
