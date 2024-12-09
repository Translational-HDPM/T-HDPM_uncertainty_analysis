#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:31:25 2023

@author: svaddadi
"""
from ensurepip import bootstrap
import os,sys,argparse
from ssl import ALERT_DESCRIPTION_DECOMPRESSION_FAILURE
from statistics import mean
import pandas as pd
import numpy as np
import random 
import scipy as sp
from scipy import stats
from scipy.stats import qmc
from scipy.optimize import fsolve
import samply 
from copy import deepcopy
from outliers import smirnov_grubbs as grubbs
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import FunctionTransformer,normalize
    
import skopt
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly
from skopt.sampler import Grid
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import openturns as ot 
import distfit
from distfit import distfit

global ADtmp 
global NCItmp
global Phenotype
global scaler


def main(argv):
    parser = argparse.ArgumentParser(description="Run George's MC simulation with different parameters")
    ### List arguements and variables and what they mean 
    parser.add_argument('-runs',dest='runs',default=100000,help = 'Number of patients being generated for each subset.')
    parser.add_argument('-int',dest='uncertainity',default=.1,help = 'Simulated uncertainity level (% of mean)')
    parser.add_argument('-user',dest='user',default='svaddadi',help = 'User of the code')
    parser.add_argument('-run',dest='runno',default=1,help = 'Iteration to set the code in')
    parser.add_argument('-filter',dest='filter',default=960,help = 'Number of samples needed to be filtered (For smallest,largest, and nboth_sides only)')
    parser.add_argument('-seed',dest='seed',default=0,help = 'Seed to set the simulations to (Default = 0)')
    parser.add_argument('-dist',dest='dist',default='normal',help = 'Distribution/Simulation to use to obtain MC values')
    parser.add_argument('-folder',dest='folder',default='MC-results-S6',help = 'Folder to Save Data In')
    parser.add_argument('-csv',dest='csv',default="../data/AD_sort_by_AD_over_NCI_v3_pop.csv",help = 'Base CSV file (Has to be in the same format as Georges file')
    parser.add_argument('-col',dest='column',default="Coeff-TPM",help = 'Column to Filter From')
    parser.add_argument('-by',dest='columnmethod',default="smallest",help = 'How to filter the Column to Filter From')
    parser.add_argument('-contamination',dest='contamination',default=.1,help = 'Contamination')
    parser.add_argument('-scaler',dest='scaler',default='StandardScaler',help = 'Type of scaler being used to normalize the data')

    parser.add_argument('-estimators',dest='estimators',default=10,help = 'Contamination')
    parser.add_argument('--bootstrap',dest='boots',action='store_true')
    parser.add_argument('--warm_start',dest='warmstart',action='store_true')
    
    parser.add_argument('-neighbors',dest='neighbors',default=10,help = 'Contamination')
    parser.add_argument('-LOF_algorithm',dest='LOF_algorithm',default='auto',help = 'Contamination')
    parser.add_argument('-leaf_size',dest='leaf_size',default=30,help = 'Contamination')
    parser.add_argument('-LOF_metric',dest='LOF_metric',default='minkowski',help = 'Contamination')

    parser.add_argument('--EE_centering',dest='EE_centering',action='store_true')
    parser.add_argument('-support_fraction',dest='support_fraction',default=None,help = 'Contamination')
    parser.add_argument('-alpha',dest='alpha',default=.05,help = 'Contamination')

    parser.add_argument('-SVM_kernel',dest='svm_kernel',default='rbf',help = 'Contamination')
    parser.add_argument('-SVM_degree',dest='svm_degree',default=3,help = 'Contamination')
    parser.add_argument('-SVM_gamma',dest='svm_gamma',default='auto',help = 'Contamination')
    parser.add_argument('-SVM_coef0',dest='svm_coef0',default=0,help = 'Contamination')
    parser.add_argument('-SVM_tol',dest='svm_coef0',default=0,help = 'Contamination')
    parser.add_argument('-SVM_nu',dest='svm_nu',default=0.05,help = 'Contamination')
    parser.add_argument('--no_shrinking',dest='shrinking',action='store_false')
    
    args=parser.parse_args()

    ### Set random seed
    random.seed(args.seed)
    

    excelsheet = pd.ExcelFile('/anvil/projects/x-cis220051/corporate/molecular-stethoscope/Teams/Team-2/ClusterMarkers_1819ADcohort-Copy1.congregated_DR.xlsx')
    # Get Sheets
    Phenotype = excelsheet.parse(excelsheet.sheet_names[0])
    TPMs = excelsheet.parse(excelsheet.sheet_names[-1])
    TPMs = TPMs[~TPMs.Coeff.isna()]
    Coeff = TPMs['Coeff'].values
    
    Phenotype = Phenotype.dropna()
    Phenotype = Phenotype.set_index('Isolate ID')

    ### Check if the folder exists and make the folder if it's not there.
    if os.path.isdir('../{}'.format(args.folder)) == False:
        os.mkdir('../{}'.format(args.folder))
    
    ### CSV File containing 965 gene dataset for AD classifier
    myDF = pd.read_csv(args.csv)
    
    myDF = myDF.set_index('Unnamed: 0')
    myDF = myDF.sort_index()

    genes = TPMs.gene_id.str.split('.',expand=True)
    genes = genes.iloc[:,0]
    TPMs['gene_id'] = genes
    TPMs = TPMs.set_index('gene_id')
    TPMs = TPMs.sort_index()
    TPMs = TPMs.loc[myDF.index.tolist(),:]
    tmp = TPMs.reset_index()
    myDF = myDF.reset_index()
    tmp = tmp.drop(['gene_id','Description', 'ORI', 'MIN', 'MAX', 'AVG', 'Coeff'],axis = 1)
    tmp = tmp.dropna(axis = 1)


    ### Basic parameters numVars represents the number of genes and numRuns is the number of scores desired 
    ### for each group in the simulation
    numVars = len(myDF.loc[:, "Coef"])
    numRuns = int(args.runs)

    ### Finding the mean TPM count for each gene in the AD and NCI groups
    ### In the dataframe, the rows of the "AD average TPM" and "NCI average TPM" columns represent 
    ### the average TPM count for a different gene for the group with Alzeheimers and the control group respectively. 
    ADmeans = myDF.loc[:, "AD average TPM"]
    Ctrlmeans = myDF.loc[:, "NCI average TPM"]
    coefficients = myDF.loc[:, "Coef"]
    if args.column == 'Coeff-TPM':
        ADcolname = 'AD Coefficent * TPM'
        NCIcolname = 'NCI coefficent * TPM'
    elif args.column == 'Avg-TPM':
        ADcolname = 'AD average TPM'
        NCIcolname = 'NCI average TPM'

    ### Filter the dataset based on the combined TPM * coeff. metric
    if int(args.filter) > 0:
        if args.columnmethod == 'smallest':
            ADmeansnsmallinds = myDF[ADcolname].nsmallest(int(args.filter)).index
            ADmeans = ADmeans.drop(ADmeansnsmallinds).reset_index(drop=True)
            #Ctrlmeansnsmallinds = myDF[NCIcolname].nsmallest(int(args.filter)).index
            #Ctrlmeans = Ctrlmeans.drop(Ctrlmeansnsmallinds).reset_index(drop=True)
            Ctrlmeans = Ctrlmeans.drop(ADmeansnsmallinds).reset_index(drop=True)
            ADmeanscoeff = coefficients.drop(ADmeansnsmallinds).reset_index(drop=True).values
            Ctrlmeanscoeff = coefficients.drop(ADmeansnsmallinds).reset_index(drop=True).values         
            #Ctrlmeanscoeff = coefficients.drop(Ctrlmeansnsmallinds).reset_index(drop=True).values
            tmp = tmp.drop(ADmeansnsmallinds).reset_index(drop=True)

        elif args.columnmethod == 'largest':
            ADmeansnsmallinds = myDF[ADcolname].nlargest(int(args.filter)).index
            ADmeans = ADmeans.drop(ADmeansnsmallinds).reset_index(drop=True)
            #Ctrlmeansnsmallinds = myDF[NCIcolname].nsmallest(int(args.filter)).index
            #Ctrlmeans = Ctrlmeans.drop(Ctrlmeansnsmallinds).reset_index(drop=True)
            Ctrlmeans = Ctrlmeans.drop(ADmeansnsmallinds).reset_index(drop=True)
            ADmeanscoeff = coefficients.drop(ADmeansnsmallinds).reset_index(drop=True).values
            Ctrlmeanscoeff = coefficients.drop(ADmeansnsmallinds).reset_index(drop=True).values         
            #Ctrlmeanscoeff = coefficients.drop(Ctrlmeansnsmallinds).reset_index(drop=True).values
            tmp = tmp.drop(ADmeansnsmallinds).reset_index(drop=True)

        
        elif args.columnmethod == 'n_bothsides':
            tempDF_AD = deepcopy(myDF)
            tempDF_NCI = deepcopy(myDF)
            ADmeansnsmallinds =tempDF_AD[ADcolname].nlargest(int(args.filter)//2).index.tolist()
            ADmeansnlargeinds =tempDF_AD[ADcolname].nsmallest(int(args.filter)//2).index.tolist()
            ADmeansinds = ADmeansnlargeinds+ADmeansnsmallinds
            ADmeans = ADmeans.drop(np.unique(np.array(ADmeansinds)),axis=0).values
            ADmeanscoeff = coefficients.drop(np.unique(np.array(ADmeansinds)),axis=0).values
            Ctrlmeansnsmallinds =tempDF_NCI[NCIcolname].nlargest(int(args.filter)//2).index.tolist()
            Ctrlmeansnlargeinds =tempDF_NCI[NCIcolname].nsmallest(int(args.filter)//2).index.tolist()
            Ctrlmeansinds = Ctrlmeansnlargeinds+Ctrlmeansnsmallinds
            #print(Ctrlmeans.index)
            #print(Ctrlmeansnsmallinds)
            Ctrlmeans = Ctrlmeans.drop(np.unique(np.array(Ctrlmeansinds)),axis=0).values
            Ctrlmeanscoeff = coefficients.drop(np.unique(np.array(Ctrlmeansinds)),axis=0).values

        elif 'Z_sigma' in args.columnmethod:
            tempDF_AD = deepcopy(myDF)
            tempDF_NCI = deepcopy(myDF)
            tempDF_AD_Z = np.abs(stats.zscore(tempDF_AD[ADcolname]))
            tempDF_NCI_Z = np.abs(stats.zscore(tempDF_AD[NCIcolname]))
            zlen = int(args.columnmethod[-1])
            ADmeans = ADmeans.drop(np.where(tempDF_AD_Z > zlen)[0]).reset_index(drop=True)
            Ctrlmeans = Ctrlmeans.drop(np.where(tempDF_NCI_Z > zlen)[0]).reset_index(drop=True)
            ADmeanscoeff = coefficients.drop(np.where(tempDF_AD_Z > zlen)[0]).values
            Ctrlmeanscoeff = coefficients.drop(np.where(tempDF_NCI_Z > zlen)[0]).values
        
        elif 'IQR' in args.columnmethod:
            Q1 = np.percentile(myDF[ADcolname], 25,interpolation = 'midpoint')
            Q3 = np.percentile(myDF[ADcolname], 75,interpolation = 'midpoint')
            IQR = Q3 - Q1
            # Upper bound
            a = float(args.columnmethod.split('_')[1])
            upper=Q3+a*IQR
            # Lower bound
            lower=Q1-a*IQR
            # Removing the outliers
            ADmeanscopy = deepcopy(ADmeans)
            ADmeans = ADmeans.drop(np.where((ADmeans > upper) | (ADmeans < lower))[0]).reset_index(drop=True).values
            ADmeanscoeff = coefficients.drop(np.where((ADmeanscopy > upper) | (ADmeanscopy < lower))[0]).reset_index(drop=True).values
            Q1 = np.percentile(myDF[NCIcolname], 25,interpolation = 'midpoint')
            Q3 = np.percentile(myDF[NCIcolname], 75,interpolation = 'midpoint')
            IQR = Q3 - Q1
            # Upper bound
            a = float(args.columnmethod.split('_')[1])
            upper=Q3+a*IQR
            # Lower bound
            lower=Q1-a*IQR
            # Removing the outliers
            Ctrlmeanscopy = deepcopy(Ctrlmeans)
            Ctrlmeans = Ctrlmeans.drop(np.where((Ctrlmeans > upper) | (Ctrlmeans < lower))[0]).reset_index(drop=True).values
            Ctrlmeanscoeff = coefficients.drop(np.where((Ctrlmeanscopy > upper) | (Ctrlmeanscopy < lower))[0]).reset_index(drop=True).values     


        elif args.columnmethod == 'Grubbs':
            ADmeans = ADmeans.drop(grubbs.two_sided_test_indices(myDF[ADcolname].values, alpha=float(args.alpha))).reset_index(drop=True)
            Ctrlmeans = Ctrlmeans.drop(grubbs.two_sided_test_indices(myDF[NCIcolname].values, alpha=float(args.alpha))).reset_index(drop=True)

            ADmeanscoeff = coefficients.drop(grubbs.two_sided_test_indices(myDF[ADcolname].values, alpha=float(args.alpha))).values
            Ctrlmeanscoeff = coefficients.drop(grubbs.two_sided_test_indices(myDF[NCIcolname].values, alpha=float(args.alpha))).values
            args.columnmethod = 'Grubbs-{}'.format(args.alpha)
        
        elif args.columnmethod == 'Grubbs-min':
            ADmeans = ADmeans.drop(grubbs.min_test_indices(myDF[ADcolname].values, alpha=float(args.alpha))).reset_index(drop=True)
            Ctrlmeans = Ctrlmeans.drop(grubbs.min_test_indices(myDF[NCIcolname].values, alpha=float(args.alpha))).reset_index(drop=True)

            ADmeanscoeff = coefficients.drop(grubbs.min_test_indices(myDF[ADcolname].values, alpha=float(args.alpha))).values
            Ctrlmeanscoeff = coefficients.drop(grubbs.min_test_indices(myDF[NCIcolname].values, alpha=float(args.alpha))).values
            args.columnmethod = 'minGrubbs-{}'.format(args.alpha)
        

        elif args.columnmethod == 'Grubbs-max':
            ADmeans = ADmeans.drop(grubbs.max_test_indices(myDF[ADcolname].values, alpha=float(args.alpha))).reset_index(drop=True)
            Ctrlmeans = Ctrlmeans.drop(grubbs.max_test_indices(myDF[NCIcolname].values, alpha=float(args.alpha))).reset_index(drop=True)

            ADmeanscoeff = coefficients.drop(grubbs.max_test_indices(myDF[ADcolname].values, alpha=float(args.alpha))).values
            Ctrlmeanscoeff = coefficients.drop(grubbs.max_test_indices(myDF[NCIcolname].values, alpha=float(args.alpha))).values
            args.columnmethod = 'maxGrubbs-{}'.format(args.alpha)
        
        elif args.columnmethod == 'IsolationForest':
            X_train = myDF[ADcolname].values.reshape(-1,1)
            iso = IsolationForest(n_estimators = int(args.estimators),contamination=float(args.contamination),bootstrap=args.boots,random_state = int(args.seed),warm_start = args.warmstart)
            yhat = iso.fit_predict(X_train)
            mask = np.where(yhat == -1)[0]
            ADmeans = ADmeans.drop(mask).reset_index(drop=True)
            ADmeanscoeff = coefficients.drop(mask).values

            X_train = myDF[NCIcolname].values.reshape(-1,1)
            iso = IsolationForest(n_estimators = int(args.estimators),contamination=float(args.contamination),bootstrap=args.boots,random_state = int(args.seed),warm_start = args.warmstart)
            yhat = iso.fit_predict(X_train)
            mask = np.where(yhat == -1)[0]
            Ctrlmeans = Ctrlmeans.drop(mask).reset_index(drop=True)
            Ctrlmeanscoeff = coefficients.drop(mask).values
            args.columnmethod = 'IsolationForest-{}-{}-{}-{}'.format(args.contamination,args.estimators,args.boots,args.warmstart)

        elif args.columnmethod == 'LocalOutlierFactor':
            X_train = myDF[ADcolname].values.reshape(-1,1)
            lof = LocalOutlierFactor(n_neighbors=args.neighbors,algorithm=args.LOF_algorithm,leaf_size=args.leaf_size,metric=args.metric)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            ADmeans = ADmeans.drop(mask).reset_index(drop=True)
            ADmeanscoeff = coefficients.drop(mask).values

            X_train = myDF[NCIcolname].values.reshape(-1,1)
            lof = LocalOutlierFactor(n_neighbors=args.neighbors,algorithm=args.LOF_algorithm,leaf_size=args.leaf_size,metric=args.metric)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            Ctrlmeans = Ctrlmeans.drop(mask).reset_index(drop=True)
            Ctrlmeanscoeff = coefficients.drop(mask).values
            args.columnmethod = 'LocalOutlierFactor-{}-{}-{}-{}'.format(args.neighbors,args.LOF_algorithm,args.leaf_size,args.metric)
        

        elif args.columnmethod == 'EllipticEnvelope':
            X_train = myDF[ADcolname].values.reshape(-1,1)
            lof = EllipticEnvelope(assume_centered=args.EE_centering,support_fraction=args.support_fraction)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            ADmeans = ADmeans.drop(mask).reset_index(drop=True)
            ADmeanscoeff = coefficients.drop(mask).values

            X_train = myDF[NCIcolname].values.reshape(-1,1)
            lof = EllipticEnvelope(assume_centered=args.EE_centering,support_fraction=args.support_fraction)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            Ctrlmeans = Ctrlmeans.drop(mask).reset_index(drop=True)
            Ctrlmeanscoeff = coefficients.drop(mask).values
            args.columnmethod = 'EllipticEnvelope-{}-{}'.format(args.EEcentering,args.support_fraction)
        
        elif args.columnmethod == 'OneClassSVM':
            X_train = myDF[ADcolname].values.reshape(-1,1)
            lof = OneClassSVM(kernel=args.svm_kernel, degree=args.svm_degree, gamma=args.svm_gamma, coef0=args.svm_coef0, tol=args.svm_tol, nu=args.svm_nu, shrinking=args.shrinking)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            ADmeans = ADmeans.drop(mask).reset_index(drop=True)
            ADmeanscoeff = coefficients.drop(mask).values

            X_train = myDF[NCIcolname].values.reshape(-1,1)
            lof = OneClassSVM(kernel=args.svm_kernel, degree=args.svm_degree, gamma=args.svm_gamma, coef0=args.svm_coef0, tol=args.svm_tol, nu=args.svm_nu, shrinking=args.shrinking)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            Ctrlmeans = Ctrlmeans.drop(mask).reset_index(drop=True)
            Ctrlmeanscoeff = coefficients.drop(mask).values
            args.columnmethod = 'OneClassSVM-{}-{}-{}-{}-{}-{}-{}'.format(args.svm_kernel,args.svm_degree,args.svm_gamma,args.svm_coef0,args.svm_tol,args.svm_nu,args.shrinking)

        
    ### Finding the standard deviation TPM count for each gene in both groups
    cpus = cpu_count()
    def Uncertainity(v):
        return float(args.uncertainity) * v
    ADstds = np.array(Parallel(n_jobs=int(cpu_count()))(delayed(Uncertainity)(val) for val in ADmeans))
    Ctrlstds = np.array(Parallel(n_jobs=int(cpu_count()))(delayed(Uncertainity)(val) for val in Ctrlmeans))
    
    ### Setting up numpy array of coefficient that represent the score function(linear combination) and adding the filter for consistency 
    ### Executing simulations for both AD group and control group
    ### Scores for Alzeheimer's experiment group
    
    if int(args.filter) > 0:
        def ADSimFcn(_,ADmeans=ADmeans, ADstds=ADstds, coeff=ADmeanscoeff,distr=args.dist):
            return Simulation(ADmeans, ADstds, coeff,distr)
    
        def CtrlSimFcn(_,Ctrlmeans=Ctrlmeans, Ctrlstds=Ctrlstds, coeff=Ctrlmeanscoeff,distr=args.dist):
            return Simulation(Ctrlmeans, Ctrlstds, coeff,distr)
        ADscores = Parallel(n_jobs=int(cpu_count()))(delayed(ADSimFcn)(val) for val in range(numRuns))
        ### Scores for Control group
        Ctrlscores = Parallel(n_jobs=int(cpu_count()))(delayed(CtrlSimFcn)(val) for val in range(numRuns))
    else:
        def ADSimFcn2(_,ADmeans=ADmeans, ADstds=ADstds, coeff=coefficients.values,distr=args.dist):
            return Simulation(ADmeans, ADstds, coeff,distr)
    
        def CtrlSimFcn2(_,Ctrlmeans=Ctrlmeans, Ctrlstds=Ctrlstds, coeff=coefficients.values,distr=args.dist):
            return Simulation(Ctrlmeans, Ctrlstds, coeff,distr)
        ADscores = Parallel(n_jobs=int(cpu_count()))(delayed(ADSimFcn2)(val) for val in range(numRuns))
        ### Scores for Control group
        Ctrlscores = Parallel(n_jobs=int(cpu_count()))(delayed(CtrlSimFcn2)(val) for val in range(numRuns))

    unscaledADdata = np.array(ADscores).reshape(args.runs,len(ADmeans))
    unscaledNCIdata = np.array(Ctrlscores).reshape(args.runs,len(ADmeans))
    unscaleddata = np.concatenate((unscaledADdata,unscaledNCIdata))   
    unscaleddata = unscaleddata.clip(min = 0)
     
    print(unscaleddata.shape)
    
    try:
        if args.scaler == 'L1':
            scaleddata = normalize(unscaleddata, norm='l1')
        elif args.scaler == 'L2':
            scaleddata = normalize(unscaleddata, norm='l2')
        elif args.scaler == 'max':
            scaleddata = normalize(unscaleddata, norm='max')            
        else:
            scaler = getattr(sklearn.preprocessing, args.scaler)().fit(tmp.T)
            scaleddata = scaler.transform(unscaleddata)
    except:
        scaler = getattr(sklearn.preprocessing, 'StandardScaler')().fit(tmp.T)
        scaleddata = scaler.transform(unscaleddata)

    scaledADdata = pd.DataFrame(scaleddata[:args.runs],columns = ['c-{}'.format(i) for i in range(len(ADmeans))])
    scaledADdata.to_csv('../{}/George-MC-sim-{}-{}-{}-{}-{}-{}-{}-{}-AD-{}-scores.csv'.format(args.folder,args.user,args.filter,args.runs,args.uncertainity,args.runno,args.dist,args.column,args.columnmethod,args.scaler))
    
    scaledADdata = scaledADdata.T
    for c in scaledADdata.columns:
        try:
            scaledADdata[c] = scaledADdata[c] * pd.Series(ADmeanscoeff)
        except: 
            scaledADdata[c] = scaledADdata[c] * coefficients
    
    scaledNCIdata = pd.DataFrame(scaleddata[args.runs:],columns = ['c-{}'.format(i) for i in range(len(ADmeans))])
    scaledNCIdata.to_csv('../{}/George-MC-sim-{}-{}-{}-{}-{}-{}-{}-{}-NCI-{}-scores.csv'.format(args.folder,args.user,args.filter,args.runs,args.uncertainity,args.runno,args.dist,args.column,args.columnmethod,args.scaler))
    scaledNCIdata = scaledNCIdata.T
    for c in scaledNCIdata.columns:
        try:
            print('Ctrl meanscoeff ',len(Ctrlmeanscoeff))
            scaledNCIdata[c] = scaledNCIdata[c] * pd.Series(Ctrlmeanscoeff)
        except:
            print('Coeff ',len(coefficients))
            scaledNCIdata[c] = scaledNCIdata[c] * coefficients

    print(scaledADdata.shape)
    ADscores = np.exp(scaledADdata.sum())/(1+np.exp(scaledADdata.sum()))
    Ctrlscores = np.exp(scaledNCIdata.sum())/(1+np.exp(scaledADdata.sum()))
    print(ADscores)
    print(Ctrlscores)
    simruns = dict()
    simruns['AD Scores'] = ADscores
    simruns['NCI Scores'] = Ctrlscores
    simruns = pd.DataFrame(simruns)
    # Finding information about distribution of the results from both simulations
    simruns.to_csv('../{}/George-MC-sim-{}-{}-{}-{}-{}-{}-{}-{}-{}-simruns.csv'.format(args.folder,args.user,args.filter,args.runs,args.uncertainity,args.runno,args.dist,args.column,args.columnmethod,args.scaler))



### The Simulation function generates a score from linear combination function based on mean and standard deviation for each gene
### In order to improve speed, I changed my score generation function from what it was in previous demos. 
### I used np.arrays and a nice trick involving a feature of np.random.normal, because numpy arrays tend to be more memory efficient 
### and speed efficient than using python lists with for loops. Using this technique greatly improved runtime. 

### It works through the means and stds arrays and one random number generated from the normal distribution formed 
### by one mean-standard deviation pair.

### Then this numpy array is multiplied by the coefficient numpy array and all of the elements in the resulting array 
### are added together to generate the final score.
def Simulation(means, stds, coefficients,fcn = 'normal'):
    ### np.random.normal(means, stds, size=(1, numVars)) creates a numpy array of "numVars" random numbers from normal distributions. 
    if fcn == 'normal':
        return np.random.normal(means, stds, size=(1, len(coefficients)))
    
    ### Provides an exponential distribution of data.   
    elif fcn == 'exponential':
        return np.random.exponential(means, size=(1, len(coefficients)))
    
    ### Provides a gamma distribution of data using only the means and standard deviations.
    elif fcn == 'gamma':
        k = means**2/stds**2
        theta = stds**2/means
        return np.random.gamma(k,theta, size=(1, len(coefficients)))
    
    ### Provides a laplace distribution of data using only the means and standard deviations.
    elif fcn == 'laplace':
        mu = means
        b = np.sqrt(stds**2/2) 
        return np.random.laplace(mu,b, size=(1, len(coefficients)))
    
    ### Provides a logistic distribution of data using only the means and standard deviations.
    elif fcn == 'logistic':
        mu = means
        s = np.sqrt(3) * stds/np.pi
        return np.random.logistic(mu,s, size=(1, len(coefficients)))
    
    ### Provides a lognormal distribution of data using only the means and standard deviations.
    elif fcn == 'lognormal':
        return np.random.lognormal(means, stds, size=(1, len(coefficients)))
    
    ### Provides a rayleigh distribution of data using only the means and standard deviations.
    elif fcn == 'rayleigh':
        sigma = np.sqrt(2/np.pi) * means
        return np.random.rayleigh(sigma, size=(1, len(coefficients)))
    
    ### Provides a Latin Hypercube sampling of data using only the means and standard deviations.
    elif fcn == 'LHS':
        sampler = qmc.LatinHypercube(d=1)
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    ### Provides a Latin Hypercube sampling of data using only the means and 2*standard deviations.
    elif fcn == 'LHS_2sigma':
        sampler = qmc.LatinHypercube(d=1)
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - 2*stds
        u_bounds = means + 2*stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Ball':
        sample = samply.ball.uniform(len(coefficients),dimensionality=1)
        ## The way scipy scales the data is (b - a) \cdot \text{sample} + a since the usual algorithm spits out dat afrom [0,1)
        ## Since the data is spit out as (-1,1) we should scale to (b-a) \cdot \text{sample} + mean 
        l_bounds = means - stds
        u_bounds = means + stds
        sample = pd.Series(sample.reshape(1,-1)[0]) 
        sample = (u_bounds-l_bounds) * sample + means
        return sample
    
    elif fcn == 'uniform-hypercube':
        sample = samply.hypercube.uniform(len(coefficients),dimensionality=1)
        l_bounds = means - stds
        u_bounds = means + stds
        sample = pd.Series(sample.reshape(1,-1)[0]) 
        sample = (u_bounds-l_bounds) * sample + means
        return sample
    
    elif fcn == 'grid-hypercube':
        sample = samply.hypercube.grid(len(coefficients),dimensionality=1)
        l_bounds = means - stds
        u_bounds = means + stds
        sample = pd.Series(sample.reshape(1,-1)[0]) 
        sample = (u_bounds-l_bounds) * sample + means
        return sample
    

    elif fcn == 'cvt-hypercube':
        sample = samply.hypercube.cvt(len(coefficients),dimensionality=1)
        l_bounds = means - stds
        u_bounds = means + stds
        sample = pd.Series(sample.reshape(1,-1)[0]) 
        sample = (u_bounds-l_bounds) * sample + means
        return sample
    
    elif fcn == 'multimodal-hypercube':
        sample = samply.hypercube.multimodal(len(coefficients),1)
        l_bounds = means - stds
        u_bounds = means + stds
        sample = pd.Series(sample.reshape(1,-1)[0]) 
        sample = (u_bounds-l_bounds) * sample + means
        return sample

    elif fcn == 'cross-shape':
        sample = samply.shape.cross(len(coefficients),1)
        l_bounds = means - stds
        u_bounds = means + stds
        sample = pd.Series(sample.reshape(1,-1)[0]) 
        sample = (u_bounds-l_bounds) * sample + means
        return sample

    elif fcn == 'curve-sample':
        sample = samply.shape.curve(len(coefficients),1)
        l_bounds = means - stds
        u_bounds = means + stds
        sample = pd.Series(sample.reshape(1,-1)[0]) 
        sample = (u_bounds-l_bounds) * sample + means
        return sample

    elif fcn == 'stripes-sample':
        sample = samply.shape.stripes(len(coefficients),1)
        l_bounds = means - stds
        u_bounds = means + stds
        sample = pd.Series(sample.reshape(1,-1)[0]) 
        sample = (u_bounds-l_bounds) * sample + means
        return sample 
    
    elif fcn == 'Sobol-1':
        sampler = qmc.Sobol(d=1)
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Sobol-2':
        sampler = qmc.Sobol(d=1,scramble=False)
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Sobol-random-1':
        sampler = qmc.Sobol(d=1,optimization='random-cd')
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Sobol-random-2':
        sampler = qmc.Sobol(d=1,scramble=False,optimization='random-cd')
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Sobol-lloyd-1':
        sampler = qmc.Sobol(d=1,optimization='lloyd')
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Sobol-lloyd-2':
        sampler = qmc.Sobol(d=1,scramble=False,optimization='lloyd')
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    

    elif fcn == 'Halton-1':
        sampler = qmc.Halton(d=1)
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Halton-2':
        sampler = qmc.Halton(d=1,scramble=False)
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Halton-random-1':
        sampler = qmc.Halton(d=1,optimization='random-cd')
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Halton-random-2':
        sampler = qmc.Halton(d=1,scramble=False,optimization='random-cd')
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Halton-lloyd-1':
        sampler = qmc.Halton(d=1,optimization='lloyd')
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims
    
    elif fcn == 'Halton-lloyd-2':
        sampler = qmc.Halton(d=1,scramble=False,optimization='lloyd')
        sample = sampler.random(n=len(coefficients))
        sample = sample.reshape(1,-1)
        l_bounds = means - stds
        u_bounds = means + stds
        sims = qmc.scale(sample, l_bounds, u_bounds)
        return sims

    elif fcn == 'Centered-LHS':
        lhs = Lhs(lhs_type="centered", criterion=None)
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Centered-maxmin-LHS':
        lhs = Lhs(lhs_type="centered", criterion='maxmin')
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'maxmin-LHS':
        lhs = Lhs(criterion='maxmin')
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Centered-correlation-LHS':
        lhs = Lhs(lhs_type="centered", criterion='correlation')
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'correlation-LHS':
        lhs = Lhs(criterion='correlation')
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Centered-ratio-LHS':
        lhs = Lhs(lhs_type="centered", criterion='ratio')
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'ratio-LHS':
        lhs = Lhs(criterion='ratio')
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims

    elif fcn == 'Hammersly':
        lhs = Hammersly()
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Grid-1':
        lhs = Grid(border="include", use_full_layout=False)
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Grid-2':
        lhs = Grid(border="exclude", use_full_layout=False)
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Grid-3':
        lhs = Grid(border="only", use_full_layout=False)
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Grid-1-full':
        lhs = Grid(border="include", use_full_layout=True)
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Grid-2-full':
        lhs = Grid(border="exclude", use_full_layout=True)
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Grid-3-full':
        lhs = Grid(border="only", use_full_layout=True)
        l_bounds = means - stds
        u_bounds = means + stds
        space = Space(list(np.concatenate((l_bounds.values.reshape(-1,1),u_bounds.values.reshape(-1,1)),axis=1)))
        n_samples = 1
        sims = np.array(lhs.generate(space.dimensions, n_samples)[0])
        return sims
    
    elif fcn == 'Reverse-Halton':
        sequence = ot.ReverseHaltonSequence(1)
        sample = np.array(sequence.generate(len(coefficients)))
        l_bounds = means - stds
        u_bounds = means + stds
        sims = (u_bounds-l_bounds) * sample.reshape(-1,1)[0] + l_bounds
        return sims

    elif fcn == 'Haselgrove':
        sequence = ot.ReverseHaltonSequence(1)
        sample = np.array(sequence.generate(len(coefficients)))
        l_bounds = means - stds
        u_bounds = means + stds
        sims = (u_bounds-l_bounds) * sample.reshape(-1,1)[0] + l_bounds
        return sims

def GetLabel(string):
    number = float(string.split('-')[0])
    return string + '-'+ Phenotype.loc[number,'Disease']

def SimFcnAD(indx):
    X = ADtmp.loc[indx,:].values
    dfit = distfit(n_boots=10)
    results = dfit.fit_transform(X)
    return dfit.model['model']

def SimFcnNCI(indx):
    X = NCItmp.loc[indx,:].values
    dfit = distfit(n_boots=10)
    results = dfit.fit_transform(X)
    return dfit.model['model']

def getRV(model):
    return model.rvs()

def DataCentricSimulationAD():
    excelsheet = pd.ExcelFile('/anvil/projects/x-cis220051/corporate/molecular-stethoscope/Teams/Team-2/ClusterMarkers_1819ADcohort-Copy1.congregated_DR.xlsx')
    # Get Sheets
    Phenotype = excelsheet.parse(excelsheet.sheet_names[0])
    TPMs = excelsheet.parse(excelsheet.sheet_names[-1])
    Coeff = TPMs['Coeff'].values
    Phenotype = Phenotype.dropna()
    Phenotype = Phenotype.set_index('Isolate ID')
    tmp = TPMs.drop(['gene_id','Description', 'ORI', 'MIN', 'MAX', 'AVG', 'Coeff'],axis = 1)
    tmpcol = tmp.columns 
    
    tmpcol = [GetLabel(stri) for stri in tmpcol]
    tmp.columns = tmpcol

    ADtmp = tmp.filter(regex='AD')
    NCItmp = tmp.filter(regex='NCI')

    models = Parallel(n_jobs=cpu_count())(delayed(SimFcnAD)(i) for i in range(len(ADtmp)))
    rvs = Parallel(n_jobs=cpu_count())(delayed(getRV)(m) for m in models)
    return np.sum(np.multiply(Coeff, rvs))

def DataCentricSimulationNCI():
    excelsheet = pd.ExcelFile('/anvil/projects/x-cis220051/corporate/molecular-stethoscope/Teams/Team-2/ClusterMarkers_1819ADcohort-Copy1.congregated_DR.xlsx')
    # Get Sheets
    Phenotype = excelsheet.parse(excelsheet.sheet_names[0])
    TPMs = excelsheet.parse(excelsheet.sheet_names[-1])
    Coeff = TPMs['Coeff'].values
    Phenotype = Phenotype.dropna()
    Phenotype = Phenotype.set_index('Isolate ID')
    tmp = TPMs.drop(['gene_id','Description', 'ORI', 'MIN', 'MAX', 'AVG', 'Coeff'],axis = 1)
    tmpcol = tmp.columns 
    
    tmpcol = [GetLabel(stri) for stri in tmpcol]
    tmp.columns = tmpcol

    ADtmp = tmp.filter(regex='AD')
    NCItmp = tmp.filter(regex='NCI')

    models = Parallel(n_jobs=cpu_count())(delayed(SimFcnNCI)(i) for i in range(len(ADtmp)))
    rvs = Parallel(n_jobs=cpu_count())(delayed(getRV)(m) for m in models)
    return np.sum(np.multiply(Coeff, rvs))









if __name__ == "__main__":
   main(sys.argv[1:])
        

