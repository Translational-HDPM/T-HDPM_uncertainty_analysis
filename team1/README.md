# The Molecular Stethoscope - Spring 2023 - Team 1


## Summary of Work
  
  The primary objective for Team 1 was to explore uncertainties in simulating raw RNA-seq data (which was TPM counts) using MC and LHS sampling techniques outlined in the Methods sections. MC simulations were run by assuming each gene TPM's for any patient is normally distributed around a mean with a pre-determined standard deviation. Random gene TPMs are then generated and scaled by Z-score. The LR classifier from the Toden et. al. paper is then utilized to obtain predicted probabilities. 200,000 patients are simulated using either MC or LHS.
   
Folder | Description 
--- | --- 
Jupyter_Notebooks | Jupyter notebooks of TPM analysis, examples of simulations done, and plots for publications and poster materials.
Python | Python functions that run the MC and LHS simulations, along with the Data-driven MC simulation.
data_files | files that have correlation and covariance matrices for the dataset. 
  
  **Jupyter_Notebooks**
  
Notebook | Description 
--- | --- 
Checking | Checks on Z-scores determined from Toden et. al. dataset and simulated MC data.
Covariance | Obtains covariance and correlation matrix. 
Distfit Analysis | Fitted distribution analysis via distfit
Filtering | Filtering of genes and seeing their effects.
Filters | checking the consistency of filtering by different subsets.
Final Analysis DD | Analysis of Data driven MC
Final Analysis | Analysis of MC and LHS
Latin-hypercube-sampling | Example of LHS
Means| Analysis of Dataset means
Metrics| Analysis of mean and std dev for all subsets. 
  
  **Python**

run-MC-sim.py -- MC and LHS simulations on the dataset. Can be run on terminal.  

To see the arguments for it. Please run: 

`python run-MC-sim.py -h`  

Data-Driven-sim.py -- MC simulations on the dataset based on fitted distributions. Can be run on terminal. 

To see the arguments for it. Please run: 

`python Data-Driven-sim.py -h`
  
 **Data Folder**
   
File | Description 
--- | --- 
AD-data-r1 | all patient data for replicate 1 for the AD subset.
AD-data-r2 | all patient data for replicate 2 for the AD subset.
AD-data | all patient data for for the AD subset.
NCI-data-r1 | all patient data for replicate 1 for the NCI subset.
NCI-data-r2 | all patient data for replicate 2 for the NCI subset.
NCI-data | all patient data for for the NCI subset.
Correlation-AD | correlation matrix for AD subset.
Correlation-NCI | correlation matrix for NCI subset.
Correlation | correlation matrix.
Covariance-AD | covariance matrix for AD subset.
Covariance-NCI | covariance matrix for NCI subset.
Covariance | covariance matrix.
Distances | bhattacharya coefficient values of all genes
Updated Toden Values | all the metrics for each gene
all-data-r1 | all patient data for replicate 1.
all-data-r2 | all patient data for replicate 2.
all-data | all patient data.

# How to Load the Conda environment 

1. Load the anaconda environment 
2. In the prompt create a conda environment using `conda activate myenv`
3. Load the environment via `conda activate myenv`
4. Update the environment using the environment.yml file using `conda env update --file environment.yml --prune`




 
# Contributors 

Name | Email 
--- | --- 
Sai Mahit Vaddadi | svaddadi@purdue.edu
Naren Ram | ram6@purdue.edu
Filip Krastev| fkrastev@asu.edu
Antonio Alejo | aalejo4@asu.edu


**License**
This project is not open source and is released under a private license. For inquiries about usage, please contact our team at:
- Members: Sai Mahit Vaddadi , John Sninsky, Dr. Mark Daniel Ward
- Email: svaddadi@purdue.edu, jsninsky@molecularstethoscope.com, mdw@purdue.edu
- Affiliations: The Data Mine @ Purdue University, Molecular Stethoscope


- Version: 1





