import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from threshold import plot_uncert_at_thresh, sub_accuracy, sub_accuracy_loess_plot
from classifier import z_score

NUM_RUNS = 100      # Number of simulation runs
UNCERTAINTY = 45    # Percentage of simulated relative standard deviation
UNCERT_RANGE = [10, 25, 50]     # Percentage RSD for V-plots
THRESH = 0.86       # Classification threshold

def main():
    myDF = pd.read_excel(
                "/anvil/projects/tdm/corporate/molecular-stethoscope/data-s23/ClusterMarkers_1819ADcohort.congregated_DR.xlsx", 
                sheet_name = 1)
    
    # setting index row name to the gene id
    myDF = myDF.set_index('gene_id')

    #Filtering out rows: discarding the ERCC rows, ERCC is a control protocol for validation of RNA sequencing
    Patients_df = myDF[~myDF.loc[:,'Coeff'].isnull()]

    # We store the coefficients(betas) of the linear classifier in an array.
    coefficients = np.nan_to_num(np.array(Patients_df.loc[:, "Coeff"]))

    # Filtering out columns with patient data
    Patients_df = Patients_df.filter(regex='^\d+')

    # group columns by patient id
    grouped_cols = Patients_df.columns.str.split('-').str[0]

    # group columns by patient id and r1/r2 suffixes
    grouped = Patients_df.groupby(grouped_cols, axis=1)

    # apply the mean function to the r1 and r2 columns for each group
    # taking mean of the replicates for subjects with multiple replicates
    Patients_df = grouped.apply(lambda x: x.mean(axis=1)).reset_index(drop=True)

    Patients_df['Mean']= Patients_df.mean(axis=1)
    Patients_df['Std'] = Patients_df.iloc[:,:-1].std(axis=1)

    # Computing and storing zscores
    Patients_df_zScore = Patients_df.apply(lambda x: z_score(x), axis=1)

    # Scatter plot of scores of simulated points versus scores of original points
    plot_uncert_at_thresh(Patients_df_zScore, coefficients, 
                        NUM_RUNS, UNCERTAINTY, THRESH)

    # V plot of agreement between simulated scores and original scores
    # sub_accuracy(Patients_df_zScore, coefficients,
    #             UNCERT_RANGE, THRESH, NUM_RUNS)

    # V plot of agreement between simulated scores and original scores,
    # with LOESS
    sub_accuracy_loess_plot(Patients_df_zScore, coefficients,
                UNCERT_RANGE, THRESH, NUM_RUNS)

if __name__ == "__main__":
    main()
