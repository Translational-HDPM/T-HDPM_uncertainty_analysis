# Team 3
### Authors: D. Tolosa, P. Joshi, R. Banda, B. Ziemann
### Purdue University, 2023.

##### I. 'Threshold and Subject-level Accuracy Analysis on RNAseq data' <br />
The main objective of the notebook is to simulate the variation for a given subject at different levels of relative standard deviation (RSD) of Transcripts Per Million (TPM) and then to demonstrate how the score varies. <br />

We have three main outputs from the notebook. <br />
Output 1: Scatter plot showing the simulated classifier scores vs the original classifier scores for all subjects. <br />
Output 2: File containing the subject-wise accuracy by calculating number of simulations that fall in FP and FN for a subject at a given threshold for further analysis. <br />
Output 3: File containing the False Positives, False Negatives, True Positives and True Negatives across 243 subjects at a given %RSD for further analysis. <br />

##### II. 'Subject-wise Uncertainty Analysis on RNAseq data' <br />
The objective of the notebook is to compute the subject-based uncertainty of the classifier. <br />
The main output from this notebook is the box plot of classifier uncertainty or standard deviation. <br />

##### III. 'PCA' <br />
In this notebook we conduct Principal Component (PC) Analysis on the TPM*Coeff data
This analysis will provide 4 outputs: <br />
Output 1. Scree Plot showing percentage of explained variance for each principal component <br />
Output 2. Scatter plot of PC1 vs PC2 for all the subjects <br />
Output 3. loading scores of the genes in different PCs <br />
Output 4. csv file of PC values for each subject <br />

### This project was developed at The Data Mine, Purdue University, in partnership with Molecular Stethoscope.


