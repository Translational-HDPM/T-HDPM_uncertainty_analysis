# Modeling Measurement Uncertainty for a Cell-Free mRNA Alzheimer’s Classifier

This repository demonstrates how to estimate and visualize measurement uncertainty for a high-dimensional RNA-Seq classifier of cell-free mRNA in Alzheimer’s disease. By using Monte Carlo simulations and regulatory guidelines, this notebook walks through the steps needed to quantify uncertainty in a clinical laboratory setting.

## Table of Contents

1. [Overview](#overview)  
2. [Background](#background)  
3. [Data and Dependencies](#data-and-dependencies)  
4. [Usage](#usage)  
5. [References](#references)  

## Overview

The goal of this repository is to

- Demonstrate best practices for estimating measurement uncertainty in molecular diagnostics.
- Incorporate site- and operator-specific variability into a published Alzheimer’s disease (AD) classification model.
- Show how Monte Carlo methods can be used to simulate sources of error and quantify overall test uncertainty.

Although the primary focus is on Alzheimer’s disease cell-free mRNA data, the workflow can be adapted for other high-dimensional classifiers where uncertainty assessment is required.

## Background

Accurate interpretation of laboratory tests depends on a clear understanding of measurement uncertainty. In a clinical diagnostic setting, having an estimate of a test’s uncertainty
- Improves interpretation of borderline or ambiguous results.
- Acts as an operational tool to detect drift or bias over time.
- Helps prioritize which analytes (genes, in this case) contribute most to total variability.
- Satisfies regulatory requirements that laboratories share uncertainty estimates with end users.

In this project, we start from an existing logistic regression model (trained to distinguish AD patients from controls using cell-free mRNA data) and explore how uncertainty propagates through that model for technical variability.

## Data and Dependencies

### Data

The notebooks assume you have access to the original cell-free mRNA expression data. If you do not, to get an idea of the general workflow please use the CSV files in the `dummy_data` folder.

> **Note:**
> 1. The raw data is not included in this repository.
> 2. The `dummy_data` files can be generated using the script `dataset_gen.py` inside `src`.
> 3. Not all notebooks have been configured to use the dummy data. Please edit the notebooks as necessary if you wish to use the dummy data with them.

### Code Dependencies

To run the notebook without modification, you’ll need at least the following Python packages:

- `python` ≥ 3.12  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`
- `scikit-learn`

Additional dependencies can be found within the `requirements.txt` file at the root level.

## Usage

### Notebooks

1. **Clone (or Download) This Repository**

```bash
git clone https://github.com/Translational-HDPM/T-HDPM_uncertainty_analysis.git
cd T-HDPM_uncertainty_analysis
```

2. **Install Dependencies**

If you’re using a virtual environment, activate it first. Then run:

```bash
pip install -r requirements.txt
```

3. **Open the Notebooks with Jupyter or JupyterLab**

Launch Jupyter or JupyterLab in this directory:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

4. **(Optional) Check out the `marimo` Versions of the Notebooks**

> **Note:**
> Only one notebook has its `marimo` version for now. Other `marimo` notebooks are under development.

Launch Marimo in this directory:

```bash
marimo edit
```

### Docker application

1. **Ensure you have all Prerequisites**

Install [Docker](https://www.docker.com/products/docker-desktop/) and clone the repository.

2. **Build the Docker Image**

Open a terminal application, navigate to the repository directory and from within the directory run.

```bash
docker build -t uncertaintyanalysisapp .
```

3. **Run the Dockerized Application**

The Docker application will run the `uncertainty_analysis_adjusted_sd.py` marimo notebook on port 2025 which can be accessed on any browser. You will need to map port 2025 to a port of your choice.

TLDR; run 
```bash
docker run --rm -p 2025:2025 uncertaintyanalysisapp
```

Then on any browser, navigate to http://localhost:2025 to view the interactive Marimo notebook app!

## References

1. Beaver *et al.*, “An FDA Perspective on the Regulatory Implications of Molecular Diagnostic Testing for Targeted Therapies,” *Clin Cancer Res.* 2017.
2. Braga & Panteghini, “The utility of measurement uncertainty in medical laboratories,” *Clin Chem Lab Med.* 2020.
3. Law *et al.*, “voom: Precision weights unlock linear model analysis tools for RNA-seq read counts,” *Genome Biol.* 2014.
4. Petraco *et al.*, “Effects of disease severity distribution on diagnostic accuracy values,” *Open Heart*.
5. Plebani *et al.*, “Measurement uncertainty: light in the shadows,” *Clin Chem Lab Med.* 2020.
6. Toden *et al.*, “Noninvasive characterization of Alzheimer’s disease using next-generation sequencing,” *Sci Adv.* 2020.

If you have suggestions for improvements or notice any errors, please submit an issue or pull request.
