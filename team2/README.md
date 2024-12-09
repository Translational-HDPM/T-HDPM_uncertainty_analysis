# Cell-Free mRNA Data Exploration and Uncertainty Analysis
The Data Mine Spring 2023
Corporate Partner: Molecular Stethoscope
Authors: Kendalyn Fruehalf, Haarika Raghavapudi, Ju Na Lee, Ruilin Yu

# Project Description
This project is part of the larger goal of uncertainty analysis in the classifier trained from cell-free mRNA (cf-mRNA) data from two patient groups (Alzheimer's Disease patients and Non-Cognitively Impaired). The data is provided to us by Molecular Stethoscope. Teams 2's sprint goal is to determine the contribution of standard deviation in RNAseq data to uncertainty in the classier. 

To accomplish this, we first visualized uncertainty in the raw data. `Team2_Sprint1-3.ipynb` reads in the cf-mRNA data and visualizes the standard deviation (SD) in Transcripts Per Million (TPM) of each gene. TPM is a measurement of mRNA level in the patient serum samples. Since FDA recommended Monte Carlo for data simulation, which works best for normally distributed data, we also visualized the mean vs SD in TPM. 

Below are some example outputs:

<img src="https://user-images.githubusercontent.com/123595447/232603343-ec5c0346-2cf1-41cc-b2b7-190c303a3a4d.png" width="500"/>
<img src="https://user-images.githubusercontent.com/123595447/232603405-0b10e903-dc8f-424d-945e-46286dbe94f0.png" width="500"/>
<img src="https://user-images.githubusercontent.com/123595447/232603599-019adc6d-e6c9-4ee7-b657-cdbcff884491.png" width="500"/>

Then, we began exploring how uncertainty in the data contributed to the classifier. `Team2_Sprint4-6.ipynb` reads in the raw data as well as data exported from `Team2_Sprint1-3.ipynb`, and generates visualizations of β\*SD and β\*mean. Some genes with large SD might have a small coefficient (β) in the classifier calculation, and thus they might not have a large contribution to the uncertainty in the classifier score. Additionally, we also visualized SD in simulated data, provided to us by Team3, which allows comparison between the classifier scores from real data, simulated data, and actual clinical diagnosis. 

Below are some example outputs:

<img src="https://user-images.githubusercontent.com/123595447/232608154-24e54724-4da9-4c54-8ce9-7e5667731d2a.png" width="500"/>
<img src="https://user-images.githubusercontent.com/123595447/232608291-f008dff2-8493-4646-8c76-b3dc48d83cf2.png" width="500"/>


# How to Run
This project was completed using R version 4.2.2 (2022-10-31) in the Anvil cluster environment. The following packages were required: readxl, ggplot2, dplyr, viridis, matrixStats. 

# Acknowledgement
We thank everyone on the Data Mine and Molecular Stethoscope team for their help throughout the project. <3
