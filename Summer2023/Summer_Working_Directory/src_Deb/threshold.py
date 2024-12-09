import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from classifier import run_sim_one_patient, cl_score, linear_score

sns.set_style("ticks")

def plot_uncert_at_thresh(Patients_df_zScore, coefficients,
                          num_runs, uncertainty, thresh):
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
    num_subj_unreliable = np.zeros(243) # keeps track of subjects whose score is unreliable under the assumed variation (0 is fine, 1 is unreliable)
    fig = plt.figure(figsize=(10,10))
    for i in range(243):
        scores = run_sim_one_patient(Patients_df_zScore.iloc[:, i], 
                                    uncertainty, 
                                    num_runs, coefficients)
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
    plt.xlabel("Classifier score", fontsize= 24)
    plt.ylabel("Simulated scores", fontsize = 24)
    plt.title('Uncertainty around threshold', fontsize = 28)
    plt.text(0.4, 0.1, 'TN', fontsize = 20)
    plt.text(0.97, 0.1, 'FN', fontsize =20)
    plt.text(0.4, 0.9, 'FP', fontsize =20)
    plt.text(0.97, 0.9, 'TP', fontsize =20)
    plt.tick_params(labelsize=18)
    plt.show()

    plt.savefig(f"uncert_around_thresh_{uncertainty}pc.png")

    return false_pos, false_neg, np.sum(num_subj_unreliable)

def sub_accuracy(Patients_df_zScore, coefficients, 
                uncertainty_range, thresh, num_runs):
    """The purpose of this function is to calculate 2 things based on %RSD (uncertainty_range), threshold (thresh), number of MC simulations (num_runs) :
    1. subject accuracy based on number of simulations : this info is provided in a dataframe called "accuracy_df"
    2. True Positives/Negatives, False Positves/Negatives: this info is provided in a dtatframe called "false_pos_df"
    
    pseudocode:
    1.  for a range of %RSD, the code starts with the first %RSD.
    2a. for each subject, it calculates the linear score by multiplying the z-scores present in dataframe "Patients_df_zScore"  by coefficient. Function used: linear_score
    2b. then it performs anti-logit operation on the linear score to get the classifier score.  Function used: cl_score 
    3.  similarly, "scores" stores the classifier score for each simulation of "num_runs"simulations, corresponding to each subject. Function used: run_sim_one_patient
    4a. since, we will plot the simulation scores against the subject scores, x_data creates the same array shape for subject scores as simulation scores.
    4b. y_data is just scores for the simplicity.
    5.  Now coming to each simuation score in each subject: we will sort out the simulation as FP, FN, TP, TN based on threshold using if-else condtions. this code will run for num_runs.
    6.  based on FP, FN, TP, TN we will calculate the accuracy.
    7.  the process 2-6 will be repeated for each subject.
    8.  the process 2-7 will be repeated for each %RSD input. 
    
    
    Example input: sub_accuracy([20, 40, 50], 0.87, 400)  -> note that uncertainty_range should be given in a form of a list even if giving single value.
    """
    real_score = np.zeros(243)
    accuracy = np.zeros(243)
    
    false_pos_df = pd.DataFrame(columns = uncertainty_range, index= ['FP', 'FN', 'TP', 'TN', 'Unreliable Subjects'])
    
    num_subj_unreliable = np.zeros(243) # keeps track of subjects whose score is unreliable under the assumed variation (0 is fine, 1 is unreliable)
    false_pos_series=[]
   
    AD = 0
    NCI = 0
    accuracy_df= pd.DataFrame(columns= uncertainty_range)
    
    for i in range(len(uncertainty_range)):
        false_pos = 0 # counter, stores number of points on second quadrant
        false_neg = 0
        true_pos = 0
        true_neg = 0
        FN = []
        FP = []
        TN = []
        TP = []
        
        for j in range(243):
            scores = run_sim_one_patient(Patients_df_zScore.iloc[:, j], 
                                        uncertainty_range[i], num_runs,
                                        coefficients) 
            y_0 = cl_score(linear_score(coefficients, Patients_df_zScore.iloc[:, j]))
            x_data = np.ones_like(scores) * y_0
            y_data = scores
            colour = np.zeros_like(x_data)
            false_neg = 0
            false_pos = 0
            for k in range(len(x_data)):
                if x_data[k] > thresh and y_data[k] < thresh:
                    false_neg = false_neg + 1
                    num_subj_unreliable[j] = 1
                elif x_data[k] < thresh and y_data[k]> thresh:
                    false_pos = false_pos + 1
                    num_subj_unreliable[j] = 1
                elif x_data[k]> thresh and y_data[k] > thresh:
                    true_pos = true_pos +1
                elif x_data[k] < thresh and y_data[k] < thresh:
                    true_neg = true_neg +1  

            accuracy[j] = (num_runs-(false_neg)-(false_pos))/num_runs
            real_score[j] = y_0

            FN.append(false_neg)
            FP.append(false_pos)
            TN.append(true_neg)
            TP.append(true_pos)
        unreliable_subjects= str(np.sum(num_subj_unreliable))
        accuracy_series = pd.Series(accuracy)
        
        sum_FN= sum(FN)
        sum_FP= sum(FP)
        sum_TN= sum(TN)
        sum_TP= sum(TP)
       
        accuracy_df[accuracy_df.columns[i]] = accuracy_series*100
        false_pos_df[false_pos_df.columns[i]]= [sum_FP, sum_FN, sum_TP, sum_TN, unreliable_subjects]
    
    Sub_score = pd.Series(real_score)
    accuracy_df['Classifier Score']= Sub_score
    dfm = accuracy_df.melt('Classifier Score', 
                var_name='%RSD', value_name='Agreement')   
    fig = sns.relplot(x= "Classifier Score" , y="Agreement", 
                      hue='%RSD', data=dfm, kind='line', 
                      height=10, aspect=1, legend=False,
                      linewidth=1, palette="rainbow")
    plt.xlabel("Classifier score", fontsize= 24)
    plt.ylabel("Agreement", fontsize = 24)
    plt.ylim(0.0, 100.0)
    plt.legend(title="% RSD", title_fontsize=18,
               labels=uncertainty_range,
               loc="best", fontsize=18)
    fig.tick_params(labelsize=18)
    fig.savefig(f"v_plot_thres_{thresh}.png")
    
    for l in range (len(Sub_score)):
        if Sub_score[l]>thresh:
            AD= AD+1
        else: 
            NCI =NCI+1
    print(f"AD = {AD}, NCI = {NCI}")
    return accuracy_df, false_pos_df

def sub_accuracy_loess_plot(Patients_df_zScore, coefficients, 
                uncertainty_range, thresh, num_runs):
    """The purpose of this function is to calculate 2 things based on %RSD (uncertainty_range), threshold (thresh), number of MC simulations (num_runs) :
    1. subject accuracy based on number of simulations : this info is provided in a dataframe called "accuracy_df"
    2. True Positives/Negatives, False Positves/Negatives: this info is provided in a dtatframe called "false_pos_df"
    
    pseudocode:
    1.  for a range of %RSD, the code starts with the first %RSD.
    2a. for each subject, it calculates the linear score by multiplying the z-scores present in dataframe "Patients_df_zScore"  by coefficient. Function used: linear_score
    2b. then it performs anti-logit operation on the linear score to get the classifier score.  Function used: cl_score 
    3.  similarly, "scores" stores the classifier score for each simulation of "num_runs"simulations, corresponding to each subject. Function used: run_sim_one_patient
    4a. since, we will plot the simulation scores against the subject scores, x_data creates the same array shape for subject scores as simulation scores.
    4b. y_data is just scores for the simplicity.
    5.  Now coming to each simuation score in each subject: we will sort out the simulation as FP, FN, TP, TN based on threshold using if-else condtions. this code will run for num_runs.
    6.  based on FP, FN, TP, TN we will calculate the accuracy.
    7.  the process 2-6 will be repeated for each subject.
    8.  the process 2-7 will be repeated for each %RSD input. 
    
    
    Example input: sub_accuracy([20, 40, 50], 0.87, 400)  -> note that uncertainty_range should be given in a form of a list even if giving single value.
    
    This function generates a V-plot based on LOESS estimates - essentially a smoother curve compared to the original.
    """
    real_score = np.zeros(243)
    accuracy = np.zeros(243)
    
    false_pos_df = pd.DataFrame(columns = uncertainty_range, 
                        index= ['FP', 'FN', 'TP', 'TN', 'Unreliable Subjects'])
    
    num_subj_unreliable = np.zeros(243) # keeps track of subjects whose score is unreliable under the assumed variation (0 is fine, 1 is unreliable)
    false_pos_series=[]
   
    AD = 0
    NCI = 0
    accuracy_df= pd.DataFrame(columns= uncertainty_range)
    accuracy_loess_df= pd.DataFrame(columns= uncertainty_range)

    for i in range(len(uncertainty_range)):
        false_pos = 0 # counter, stores number of points on second quadrant
        false_neg = 0
        true_pos = 0
        true_neg = 0
        FN = []
        FP = []
        TN = []
        TP = []
        
        for j in range(243):
            scores = run_sim_one_patient(Patients_df_zScore.iloc[:, j], 
                                        uncertainty_range[i], num_runs,
                                        coefficients) 
            y_0 = cl_score(linear_score(coefficients, 
                                    Patients_df_zScore.iloc[:, j]))
            x_data = np.ones_like(scores) * y_0
            y_data = scores
            colour = np.zeros_like(x_data)
            false_neg = 0
            false_pos = 0
            for k in range(len(x_data)):
                if x_data[k] > thresh and y_data[k] < thresh:
                    false_neg = false_neg + 1
                    num_subj_unreliable[j] = 1
                elif x_data[k] < thresh and y_data[k]> thresh:
                    false_pos = false_pos + 1
                    num_subj_unreliable[j] = 1
                elif x_data[k]> thresh and y_data[k] > thresh:
                    true_pos = true_pos +1
                elif x_data[k] < thresh and y_data[k] < thresh:
                    true_neg = true_neg +1  

            accuracy[j] = (num_runs-(false_neg)-(false_pos))/num_runs
            real_score[j] = y_0

            FN.append(false_neg)
            FP.append(false_pos)
            TN.append(true_neg)
            TP.append(true_pos)
        unreliable_subjects= str(np.sum(num_subj_unreliable))
        accuracy_series = pd.Series(accuracy)
        
        sum_FN= sum(FN)
        sum_FP= sum(FP)
        sum_TN= sum(TN)
        sum_TP= sum(TP)
       
        accuracy_df[accuracy_df.columns[i]] = accuracy_series*100
        if uncertainty_range[i] < 15:
            accuracy_loess_df[accuracy_df.columns[i]] = accuracy_series*100
        else:
            accuracy_loess_df[accuracy_df.columns[i]] = \
                        lowess(accuracy, 
                                real_score,
                                frac=0.15,
                                return_sorted=False) *100
        false_pos_df[false_pos_df.columns[i]]= [sum_FP, sum_FN, 
                                        sum_TP, sum_TN, unreliable_subjects]
    
    Sub_score = pd.Series(real_score)
    accuracy_df['Classifier Score']= Sub_score
    accuracy_loess_df['Classifier Score'] = Sub_score
    
    dfm = accuracy_loess_df.melt('Classifier Score', 
                var_name='%RSD', value_name='Agreement')   
    fig = sns.relplot(x= "Classifier Score" , y="Agreement", 
                      hue='%RSD', data=dfm, kind='line', 
                      height=10, aspect=1, legend=False,
                      linewidth=1, palette="rainbow")
    plt.xlabel("Classifier score", fontsize= 24)
    plt.ylabel("Agreement", fontsize = 24)
    plt.ylim(0.0, 100.0)
    plt.legend(title="% RSD", title_fontsize=18,
               labels=uncertainty_range,
               loc="best", fontsize=18)
    fig.tick_params(labelsize=18)
    fig.savefig(f"v_plot_loess_thres_{thresh}.png")
    
    for l in range (len(Sub_score)):
        if Sub_score[l]>thresh:
            AD= AD+1
        else: 
            NCI =NCI+1
    print(f"AD = {AD}, NCI = {NCI}")
    return accuracy_df, false_pos_df
