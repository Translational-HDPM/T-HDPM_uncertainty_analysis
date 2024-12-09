import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Reding csv file containing 243 subjects and their raw TPM counts across 1059 genes
### Source of data
myDF = pd.read_excel("/anvil/projects/tdm/corporate/molecular-stethoscope/data-s23/ClusterMarkers_1819ADcohort.congregated_DR.xlsx", sheet_name = 1)
# setting index row name to the gene id
myDF = myDF.set_index('gene_id')
myDF2 = pd.read_excel("/anvil/projects/tdm/corporate/molecular-stethoscope/data-s23/ClusterMarkers_1819ADcohort.congregated_DR.xlsx", sheet_name = 0)

# len(myDF.columns)

###Step 1: Data prep

#Filtering out rows: discarding the ERCC rows, ERCC is a control protocol for validation of RNA sequencing
Patients_df = myDF[~myDF.loc[:,'Coeff'].isnull()]

# We store the coefficients(betas) of the linear classifier in an array.
coefficients = np.nan_to_num(np.array(Patients_df.loc[:, "Coeff"]))

# Filtering out columns with patient data
Patients_df = Patients_df.filter(regex='^\d+')

#Patients_df

#filter the replicate you want to use
Patient_r1= Patients_df.filter(regex= 'r1')
len(Patient_r1.columns)

def rename_col(x):
    x = x.split('-')[0]
    return x

Patient_r1= Patient_r1.rename(columns= rename_col)

Patient_r1.head()

patient_id= list(Patient_r1.columns.values)
patient_id= list(map(int,patient_id))
#patient_id

#Patient_r1.drop(Patient_r1["11182"]<= 10, inplace = True)

# # group columns by patient id
# grouped_cols = Patients_df.columns.str.split('-').str[0]

# # group columns by patient id and r1/r2 suffixes
# grouped = Patients_df.groupby(grouped_cols, axis=1)

# # apply the mean function to the r1 and r2 columns for each group
# # taking mean of the replicates for subjects with multiple replicates
# Patients_df = grouped.apply(lambda x: x.mean(axis=1)).reset_index(drop=True)
# #Patients_df.head()

###Step 2: Computing Zscores and RSD from TPM data


Patient_r1['Mean']= Patient_r1.mean(axis=1)
Patient_r1['Std']=Patient_r1.iloc[:,:-1].std(axis=1)
Patient_r1['RSD'] = (Patient_r1['Std'] / Patient_r1['Mean'])*100 # New code Filip

Patient_r1.head()
#print(Patients_df.shape)

rsd= Patient_r1["RSD"]
rsd= list(rsd)
rsd[0]

sum(float(num) >=50  for num in rsd)

# We define a function whose input is TPM and outputs the corresponding Zscore
def z_score(x):
    return (x-x['Mean'])/x['Std']

# Computing and storing zscores
Patients_df_zScore = Patient_r1.apply(lambda x: z_score(x), axis=1)
Patients_df_zScore.head()
#print(Patients_df_zScore.shape)

Patient_r1= Patient_r1.set_index([ 'Mean', 'Std', 'RSD'])    
Patient_r1.head()

#Patient_r1.to_csv('R2_data.csv')


patient_id= list(Patient_r1.columns.values)
patient_id= list(map(int,patient_id))
#patient_id

#Patient_r1.drop(Patient_r1["11182"]<= 10, inplace = True)

# # group columns by patient id
# grouped_cols = Patients_df.columns.str.split('-').str[0]

# # group columns by patient id and r1/r2 suffixes
# grouped = Patients_df.groupby(grouped_cols, axis=1)

# # apply the mean function to the r1 and r2 columns for each group
# # taking mean of the replicates for subjects with multiple replicates
# Patients_df = grouped.apply(lambda x: x.mean(axis=1)).reset_index(drop=True)
# #Patients_df.head()

###Step 2: Computing Zscores and RSD from TPM data


Patient_r1['Mean']= Patient_r1.mean(axis=1)
Patient_r1['Std']=Patient_r1.iloc[:,:-1].std(axis=1)
Patient_r1['RSD'] = (Patient_r1['Std'] / Patient_r1['Mean'])*100 # New code Filip

Patient_r1.head()
#print(Patients_df.shape)

rsd= Patient_r1["RSD"]
rsd= list(rsd)
rsd[0]

sum(float(num) >=50  for num in rsd)

# We define a function whose input is TPM and outputs the corresponding Zscore
def z_score(x):
    return (x-x['Mean'])/x['Std']

# Computing and storing zscores
Patients_df_zScore = Patient_r1.apply(lambda x: z_score(x), axis=1)
Patients_df_zScore.head()
#print(Patients_df_zScore.shape)

Patient_r1= Patient_r1.set_index([ 'Mean', 'Std', 'RSD'])    
Patient_r1.head()

#Patient_r1.to_csv('R2_data.csv')

### Step 3: Defining Monte Carlo simulation and classifier functions using RSD

#np.random.seed(46215423)

#SImulation using gene-specific RSD
def RSD_simulation(subject, rsd, coefficient):
    score=0
    subject_list = list(subject)
    for i in range (len(subject_list)):
        std= rsd[i]/100 * subject_list[i]
        std = np.abs(std)
        sum_number= (coefficients[i]) * int(np.random.normal(subject_list[i], std, size=1))
        score= score+ sum_number
    return(score)
        
# Sampling function performing the Monte Carlo simulations
def Simulation(means, rsd, coefficients):
    std= [rsd/100 * val for val in means]
    std = np.abs(std)
    return np.sum(np.multiply(coefficients, np.random.normal(means, std, size=(1, len(coefficients)))))

# Function to perform anti-logit operation on the linear score 
def cl_score(linear_score, gamma = 0):
    temp = gamma + linear_score
    classifier_score = np.exp(temp) / (1 + np.exp(temp))
    return classifier_score

# Function to calculate subject wise mean and rsd of simulated scores 
def run_sim_one_patient_mean_sd(col):
    temp_Sim = [RSD_simulation(col, rsd, coefficients) for _ in range(numRuns)]
    return [np.mean(temp_Sim),np.std(temp_Sim)]

# Function to calculate the classifier score for each simulation of "num_runs" simulations, corresponding to each subject
def run_sim_one_patient(col, num_runs):
    temp_Sim = np.asarray([cl_score(RSD_simulation(col, rsd, coefficients)) for _ in range(num_runs)])
    return temp_Sim

# This score is the classifer linear score we want to compare with the simulated scores
def linear_score(coefficients, col):
    linear_score = np.sum(coefficients * col, axis=0)
    return linear_score

### Step 4: (Hyperparameters) We decide on number of simulations per subject (num_runs), assumed variation on TPM counts (uncertainty), and a classification threshold (thresh) ---¶ num_runs is the number of scores we will generate per patient, each score is generated by choosing a TPM value from a normal distribution that has the actual value as mean (or the average of the TPM counts if there are replicates) and rsd determined by the variable 'uncertainty'.

num_runs = 1
uncertainty = 25
thresh = 0.04874941
num_runs = 1
uncertainty = 25
thresh = 0.04874941
Step 5: Defining a function that plots the uncertainty around the threshold, and returns the figure, false positive count, false negative count, and number of subjects with unreliable classification for the specified variation and threshold.

def plot_uncert_at_thresh(num_runs, thresh ):
    
    
    """The purpose of this function is to produce 2 things based on %RSD (uncertainty), threshold (thresh), number of MC simulations (num_runs) :
    1. Figure to show the scattering based on classifier scores of simulations against subject.
    2. False Positves/Negatives
    
    pseudocode:
    1.  for a range of %RSD, the code starts with the first %RSD.
    2a. for each subject, it calculates the linear score by multiplying the z-scores present in dataframe "Patients_df_zScore"  by coefficient. Function used: linear_score
    2b. then it performs anti-logit operation on the linear score to get the classifier score.  Function used: cl_score 
    3.  similarly, "scores" stores the classifier score for each simulation of "num_runs"simulations, corresponding to each subject. Function used: run_sim_one_patient
    4a. since, we will plot the simulation scores against the subject scores, x_data creates the same array shape for subject scores as simulation scores.
    4b. y_data is just scores for the simplicity.
    5.  Now coming to each simuation score in each subject: we will sort out the simulation as FP, FN, TP, TN based on threshold using if-else condtions. this code 
        will run for num_runs.
    6.  based on FP, FN, TP, TN we will calculate the accuracy.
    7.  the process 2-6 will be repeated for each subject.
    8.  A scatter plot will be produced using simulation scores on y-axis and subject scores on x-axis, two perpendicular lines passing the thresholds on x and y axis
     will  categorize the scatter dots into FP, FN , TP and TN.
    
    
    Example input: sub_accuracy(400, 50, 0.87)
    """
    
    false_pos = 0 # counter, stores number of points on second quadrant
    false_neg = 0
    num_subj_unreliable = np.zeros(len(Patient_r1.columns)) # keeps track of subjects whose score is unreliable under the assumed variation (0 is fine, 1 is unreliable)
    fig = plt.figure(figsize=(10,10))
    #for i in range(243):
    for i in range(len(Patient_r1.columns)):
        scores = run_sim_one_patient(Patients_df_zScore.iloc[:, i], num_runs)
        y_0 = cl_score(linear_score(coefficients, Patients_df_zScore.iloc[:, i]))
        x_data = np.ones_like(scores) * y_0
        y_data = scores
        colour = np.zeros_like(x_data)
        for j in range(len(x_data)):
            if x_data[j] > thresh and y_data[j] < thresh:
                colour[j] = 1
                false_neg = false_neg + 1
                num_subj_unreliable[i] = 1
            elif x_data[j] < thresh and y_data[j]> thresh:
                colour[j] = 2
                false_pos = false_pos + 1
                num_subj_unreliable[i] = 1
        plt.scatter(x_data, y_data, c=colour, cmap = 'Dark2_r' ,alpha = 0.15, s=100)
    plt.axvline(x = thresh, color = 'g', linestyle= '--')
    plt.axhline(y = thresh, color = 'g', linestyle= '--')
    #plt.xlim([0.425, 0.575])
    #plt.ylim([0.425, 0.575])
    plt.xlabel("Classifier score", fontsize= 24)
    plt.ylabel("Simulated scores", fontsize = 24)
    plt.title('Uncertainty around threshold', fontsize = 28)
    # Below are the labels of agreement and disagreement
    A = plt.scatter([0],[0], alpha =0) # dummy plot for legend
    B = plt.scatter([0],[0],  alpha =0) # dummy plot for legend
    plt.text(0, 0.8, 'A', fontsize = 20)
    plt.text(0, 0, 'B', fontsize =20)
    plt.text(0.8, 0, 'B', fontsize =20)
    plt.text(0.8, 0.8, 'A', fontsize =20)
    plt.legend([A,B], ['A : Agreement', 'B : Disagreement'])
    #plt.savefig('figures/uncert_around_thresh_25_RSD.png')

    return [ fig , false_pos , false_neg , (np.sum(num_subj_unreliable)) ]

### Check that random seeding is working by running this cell several times... it is working.
np.random.seed(78987765)
scores = run_sim_one_patient(Patients_df_zScore.iloc[:, 67], 10)
sorted(scores)

plot_uncert_at_thresh(1, thresh)