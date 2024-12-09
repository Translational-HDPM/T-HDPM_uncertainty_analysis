# Import System Libraries 
from ensurepip import bootstrap
import os,sys,argparse
from ssl import ALERT_DESCRIPTION_DECOMPRESSION_FAILURE
from copy import deepcopy

# Import Data Libraries 
from statistics import mean
import pandas as pd
import numpy as np
import random 
from scipy import stats

# Import  Filtering Libraries 
from outliers import smirnov_grubbs as grubbs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

# Import Parallel Libraries 
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# Import Pre-processing Library for Z Score
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import normalize

# Import Disfit for Statistical Fitting 
import distfit
from distfit import distfit

#Global Variables
global ADtmp 
global NCItmp
global Phenotype
global ADcoeff
global NCIcoeff


# Function that Fits a Distribution to a Gene for AD patients
# Args: 
# - indx --> row that has the gene TPM values 
def SimFcnAD(indx):
    X = ADtmp.loc[indx,:].values
    dfit = distfit(n_boots=10)
    results = dfit.fit_transform(X)
    return dfit.model['model']

# Function that Fits a Distribution to a Gene for NCI patients
# Args: 
# - indx --> row that has the gene TPM values 
def SimFcnNCI(indx):
    X = NCItmp.loc[indx,:].values
    dfit = distfit(n_boots=10)
    results = dfit.fit_transform(X)
    return dfit.model['model']


# Function that Simulates Random Variables for a given distribution
# Args: 
# - model --> distfit model
def getRV(model):
    return model.rvs()


# Function that simulates TPM counts for a the AD data
def DataCentricSimulationAD():
    models = Parallel(n_jobs=cpu_count())(delayed(SimFcnAD)(i) for i in range(len(ADtmp)))
    rvs = Parallel(n_jobs=cpu_count())(delayed(getRV)(m) for m in models)
    return np.multiply(ADcoeff, rvs)

# Function that simulates TPM counts for a the NCI data
def DataCentricSimulationNCI():
    models = Parallel(n_jobs=cpu_count())(delayed(SimFcnNCI)(i) for i in range(len(NCItmp)))
    rvs = Parallel(n_jobs=cpu_count())(delayed(getRV)(m) for m in models)
    return np.multiply(NCIcoeff, rvs)

def main(argv):
    parser = argparse.ArgumentParser(description="Run George's MC simulation with different parameters")
    
    ### List arguements and variables and what they mean 
    parser.add_argument('-runs',dest='runs',default=100000,help = 'MC runs')
    parser.add_argument('-user',dest='user',default='svaddadi',help = 'MC user')
    parser.add_argument('-run',dest='runno',default=1,help = 'MC run no')
    parser.add_argument('-filter',dest='filter',default=0,help = 'How many of the smallest values we want to filter')
    parser.add_argument('-seed',dest='seed',default=0,help = 'Seed to set the simulations to (Default = 0)')
    parser.add_argument('-folder',dest='folder',default='MC-results-S2',help = 'Folder to Save Data In')
    parser.add_argument('-csv',dest='csv',default="../data/AD_sort_by_AD_over_NCI_v3_pop.csv",help = 'CSV File to Read')
    parser.add_argument('-col',dest='column',default="Coeff-AVG",help = 'Column to Filter From')
    parser.add_argument('-by',dest='columnmethod',default="smallest",help = 'How to filter the Column to Filter From')
    parser.add_argument('-contamination',dest='contamination',default=.1,help = 'Contamination')
    parser.add_argument('-scaler',dest='scaler',default='StandardScaler',help = 'Scikit-learn scaler to use')

    parser.add_argument('-estimators',dest='estimators',default=10,help = 'Estimators for LOF')
    parser.add_argument('--bootstrap',dest='boots',action='store_true')
    parser.add_argument('--warm_start',dest='warmstart',action='store_true')
    
    parser.add_argument('-neighbors',dest='neighbors',default=10,help = 'Neighbors for LOF')
    parser.add_argument('-LOF_algorithm',dest='LOF_algorithm',default='auto',help = 'LOF algorithm')
    parser.add_argument('-leaf_size',dest='leaf_size',default=30,help = 'Leaf size for LOF')
    parser.add_argument('-LOF_metric',dest='LOF_metric',default='minkowski',help = 'LOF metric')

    parser.add_argument('--EE_centering',dest='EE_centering',action='store_true')
    parser.add_argument('-support_fraction',dest='support_fraction',default=None,help = 'Support Fraction for EE')
    parser.add_argument('-alpha',dest='alpha',default=.05,help = 'Alpha for EE')

    parser.add_argument('-SVM_kernel',dest='svm_kernel',default='rbf',help = 'Kernel for SVM')
    parser.add_argument('-SVM_degree',dest='svm_degree',default=3,help = 'SVM degree')
    parser.add_argument('-SVM_gamma',dest='svm_gamma',default='auto',help = 'gamma coeff for SVM')
    parser.add_argument('-SVM_coef0',dest='svm_coef0',default=0,help = 'coeff for SVM')
    parser.add_argument('-SVM_tol',dest='svm_tol',default=0,help = 'Tolerance for SVM')
    parser.add_argument('-SVM_nu',dest='svm_nu',default=0.05,help = 'Nu for SVM')
    parser.add_argument('--no_shrinking',dest='shrinking',action='store_false')

    args=parser.parse_args()

    ### Check if the folder exists and make the folder if it's not there.
    if os.path.isdir('../{}'.format(args.folder)) == False:
        os.mkdir('../{}'.format(args.folder))
    
    
    # Import Raw Data 
    excelsheet = pd.ExcelFile('/anvil/projects/x-cis220051/corporate/molecular-stethoscope/Teams/Team-2/ClusterMarkers_1819ADcohort-Copy1.congregated_DR.xlsx')
    # Get Sheets
    Phenotype = excelsheet.parse(excelsheet.sheet_names[0])
    TPMs = excelsheet.parse(excelsheet.sheet_names[-1])
    # Filter out NA values and get Coefficients 
    TPMs = TPMs[~TPMs.Coeff.isna()]
    Coeff = TPMs['Coeff'].values
    
    # Filter out Phenotype 
    Phenotype = Phenotype.dropna()
    Phenotype = Phenotype.set_index('Isolate ID')

    # Drop unnecessary columns for simulation
    tmp = TPMs.drop(['gene_id','Description', 'ORI', 'MIN', 'MAX', 'AVG', 'Coeff'],axis = 1)
    tmpcol = tmp.columns 
    
    # Drop labels
    def GetLabel(string):
        number = float(string.split('-')[0])
        return string + '-'+ Phenotype.loc[number,'Disease']

    # Get labels
    tmpcol = [GetLabel(stri) for stri in tmpcol]
    tmp.columns = tmpcol

    # Get Bhattacharya distance Matrix
    distances = pd.read_csv('/anvil/projects/x-cis220051/corporate/molecular-stethoscope/Teams/Team-1/monte-carlo-demos/submits/Distances.csv')
       
    # Get AD and NCI subsets
    ADtmp = tmp.filter(regex='AD')
    NCItmp = tmp.filter(regex='NCI')
    
    #Check if the column being filtered by is based on the Beta or Not 
    if 'Coeff' in args.column:
        #Factor in the Beta
        tmp2 = tmp
        for col in tmp.columns: 
            tmp2[col] = tmp[col] * Coeff
        # Get AD and NCI subsets
        ADtmp2 = tmp2.filter(regex='AD')
        NCItmp2 = tmp2.filter(regex='NCI')
        
        # Get Metrics
        AVG = tmp2.mean(axis=1)
        STD = tmp2.std(axis=1)
        MED = tmp2.median(axis=1)
        MAX = tmp2.max(axis=1)
        BHATT = distances.bhattadnci

        #Save metrics to AD and NCI dataframes
        ADtmp['AVG'] = AVG
        ADtmp['STD'] = STD
        ADtmp['MED'] = MED
        ADtmp['MAX'] = MAX
        ADtmp['BHATT'] = BHATT
        
        NCItmp['AVG'] = AVG
        NCItmp['STD'] = STD
        NCItmp['MED'] = MED
        NCItmp['MAX'] = MAX
        NCItmp['BHATT'] = BHATT
        
    else:
        # Get Metrics
        AVG = tmp.mean(axis=1)
        STD = tmp.std(axis=1)
        MED = tmp.median(axis=1)
        MAX = tmp.max(axis=1)
        BHATT = distances.bhattadnci
        #Save metrics to AD and NCI dataframes
        ADtmp['AVG'] = AVG
        ADtmp['STD'] = STD
        ADtmp['MED'] = MED
        ADtmp['MAX'] = MAX
        ADtmp['BHATT'] = BHATT
        
        NCItmp['AVG'] = AVG
        NCItmp['STD'] = STD
        NCItmp['MED'] = MED
        NCItmp['MAX'] = MAX
        NCItmp['BHATT'] = BHATT

    #Save Coefficients into Dataframe
    ADtmp['C'] = Coeff
    NCItmp['C'] = Coeff

    
    # Convert the Columns stated in argparse to Columns
    if 'AVG' in args.column:
        colname = 'AVG'
    elif 'STD' in args.column:
        colname = 'STD'
    elif 'MED' in args.column:
        colname = 'MED'
    elif 'MAX' in args.column:
        colname = 'MAX'
    elif 'BHATT' in args.column:
        colname = 'BHATT'

        
    #Filter the data if needed
    if int(args.filter) > 0:
        if args.columnmethod == 'smallest':
            # Get Smallest TPM or Beta TPM counts
            ADmeansnsmallinds = ADtmp[colname].nsmallest(int(args.filter)).index
            # Filter both sets of data by that
            ADtmp = ADtmp.drop(ADmeansnsmallinds).reset_index(drop=True)
            NCItmp = NCItmp.drop(ADmeansnsmallinds).reset_index(drop=True)
            
        elif args.columnmethod == 'largest':
            # Get largest TPM or Beta TPM counts
            ADmeansnsmallinds = ADtmp[colname].nlargest(int(args.filter)).index\
            # Filter both sets of data by that
            ADtmp = ADtmp.drop(ADmeansnsmallinds).reset_index(drop=True)
            NCItmp = NCItmp.drop(ADmeansnsmallinds).reset_index(drop=True)
            
        elif args.columnmethod == 'n_bothsides':
            # Copy Data 
            tempDF_AD = deepcopy(ADtmp)
            # Get TPM indices
            ADmeansnsmallinds =tempDF_AD[colname].nlargest(int(args.filter)//2).index.tolist()
            ADmeansnlargeinds =tempDF_AD[colname].nsmallest(int(args.filter)//2).index.tolist()
            ADmeansinds = ADmeansnlargeinds+ADmeansnsmallinds
            # Filter indices
            ADtmp = ADtmp.drop(np.unique(np.array(ADmeansinds)),axis=0).values
            NCItmp = NCItmp.drop(np.unique(np.array(ADmeansinds)),axis=0).values
            
        elif 'Z_sigma' in args.columnmethod:
            tempDF_AD = deepcopy(ADtmp)
            tempDF_NCI = deepcopy(NCItmp)
            tempDF_AD_Z = np.abs(stats.zscore(tempDF_AD[colname]))
            tempDF_NCI_Z = np.abs(stats.zscore(tempDF_NCI[colname]))
            zlen = int(args.columnmethod[-1])
            ADtmp = ADtmp.drop(np.where(tempDF_AD_Z > zlen)[0]).reset_index(drop=True)
            NCItmp = NCItmp.drop(np.where(tempDF_NCI_Z > zlen)[0]).reset_index(drop=True)
            
        elif 'IQR' in args.columnmethod:
            Q1 = np.percentile(ADtmp[colname], 25,interpolation = 'midpoint')
            Q3 = np.percentile(ADtmp[colname], 75,interpolation = 'midpoint')
            IQR = Q3 - Q1
            # Upper bound
            a = float(args.columnmethod.split('_')[1])
            upper=Q3+a*IQR
            # Lower bound
            lower=Q1-a*IQR
            # Removing the outliers
            ADtmp2 = ADtmp2.drop(np.where((ADtmp > upper) | (ADtmp < lower))[0]).reset_index(drop=True).values
            Q1 = np.percentile(NCItmp[colname], 25,interpolation = 'midpoint')
            Q3 = np.percentile(NCItmp[colname], 75,interpolation = 'midpoint')
            IQR = Q3 - Q1
            # Upper bound
            a = float(args.columnmethod.split('_')[1])
            upper=Q3+a*IQR
            # Lower bound
            lower=Q1-a*IQR
            # Removing the outliers
            NCItmp = NCItmp.drop(np.where((NCItmp > upper) | (NCItmp < lower))[0]).reset_index(drop=True).values
            
        elif args.columnmethod == 'Grubbs':
            ADtmp = ADtmp.drop(grubbs.two_sided_test_indices(ADtmp[colname].values, alpha=float(args.alpha))).reset_index(drop=True)
            NCItmp = NCItmp.drop(grubbs.two_sided_test_indices(NCItmp[colname].values, alpha=float(args.alpha))).reset_index(drop=True)
            args.columnmethod = 'Grubbs-{}'.format(args.alpha)
        
        elif args.columnmethod == 'Grubbs-min':
            ADtmp = ADtmp.drop(grubbs.min_test_indices(ADtmp[colname].values, alpha=float(args.alpha))).reset_index(drop=True)
            NCItmp = NCItmp.drop(grubbs.min_test_indices(ADtmp[colname].values, alpha=float(args.alpha))).reset_index(drop=True)
            args.columnmethod = 'minGrubbs-{}'.format(args.alpha)
        
        elif args.columnmethod == 'Grubbs-max':
            ADtmp = ADtmp.drop(grubbs.max_test_indices(ADtmp[colname].values, alpha=float(args.alpha))).reset_index(drop=True)
            NCItmp = NCItmp.drop(grubbs.max_test_indices(NCItmp[colname].values, alpha=float(args.alpha))).reset_index(drop=True)
            args.columnmethod = 'maxGrubbs-{}'.format(args.alpha)
        
        elif args.columnmethod == 'IsolationForest':
            X_train = ADtmp[colname].values.reshape(-1,1)
            iso = IsolationForest(n_estimators = int(args.estimators),contamination=float(args.contamination),bootstrap=args.boots,random_state = int(args.seed),warm_start = args.warmstart)
            yhat = iso.fit_predict(X_train)
            mask = np.where(yhat == -1)[0]
            ADtmp = ADtmp.drop(mask).reset_index(drop=True)
            
            X_train = NCItmp[colname].values.reshape(-1,1)
            iso = IsolationForest(n_estimators = int(args.estimators),contamination=float(args.contamination),bootstrap=args.boots,random_state = int(args.seed),warm_start = args.warmstart)
            yhat = iso.fit_predict(X_train)
            mask = np.where(yhat == -1)[0]
            NCItmp = NCItmp.drop(mask).reset_index(drop=True)
            args.columnmethod = 'IsolationForest-{}-{}-{}-{}'.format(args.contamination,args.estimators,args.boots,args.warmstart)

        elif args.columnmethod == 'LocalOutlierFactor':
            X_train = ADtmp[colname].values.reshape(-1,1)
            lof = LocalOutlierFactor(n_neighbors=args.neighbors,algorithm=args.LOF_algorithm,leaf_size=args.leaf_size,metric=args.metric)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            ADtmp = ADtmp.drop(mask).reset_index(drop=True)
            
            X_train = NCItmp[colname].values.reshape(-1,1)
            lof = LocalOutlierFactor(n_neighbors=args.neighbors,algorithm=args.LOF_algorithm,leaf_size=args.leaf_size,metric=args.metric)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            NCItmp = NCItmp.drop(mask).reset_index(drop=True)
            args.columnmethod = 'LocalOutlierFactor-{}-{}-{}-{}'.format(args.neighbors,args.LOF_algorithm,args.leaf_size,args.metric)
        

        elif args.columnmethod == 'EllipticEnvelope':
            X_train = ADtmp[colname].values.reshape(-1,1)
            lof = EllipticEnvelope(assume_centered=args.EE_centering,support_fraction=args.support_fraction)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            ADtmp = ADtmp.drop(mask).reset_index(drop=True)
            
            X_train = NCItmp[colname].values.reshape(-1,1)
            lof = EllipticEnvelope(assume_centered=args.EE_centering,support_fraction=args.support_fraction)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            NCItmp = NCItmp.drop(mask).reset_index(drop=True)
            args.columnmethod = 'EllipticEnvelope-{}-{}'.format(args.EEcentering,args.support_fraction)
        
        elif args.columnmethod == 'OneClassSVM':
            X_train = ADtmp2[colname].values.reshape(-1,1)
            lof = OneClassSVM(kernel=args.svm_kernel, degree=args.svm_degree, gamma=args.svm_gamma, coef0=args.svm_coef0, tol=args.svm_tol, nu=args.svm_nu, shrinking=args.shrinking)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            ADtmp = ADtmp.drop(mask).reset_index(drop=True)
            
            X_train = NCItmp[colname].values.reshape(-1,1)
            lof = OneClassSVM(kernel=args.svm_kernel, degree=args.svm_degree, gamma=args.svm_gamma, coef0=args.svm_coef0, tol=args.svm_tol, nu=args.svm_nu, shrinking=args.shrinking)
            yhat = lof.fit_predict(X_train)
            # select all rows that are not outliers
            mask = np.where(yhat == -1)
            NCItmp = NCItmp.drop(mask).reset_index(drop=True)
            args.columnmethod = 'OneClassSVM-{}-{}-{}-{}-{}-{}-{}'.format(args.svm_kernel,args.svm_degree,args.svm_gamma,args.svm_coef0,args.svm_tol,args.svm_nu,args.shrinking)

    # Get Coefficients
    ADcoeff = ADtmp.C
    NCIcoeff = NCItmp.C

    # Drop unneeded columns for further analysis
    ADtmp = ADtmp.drop(['AVG','STD','MED','MAX','BHATT','C'],axis = 1)
    NCItmp = NCItmp.drop(['AVG','STD','MED','MAX','BHATT','C'],axis = 1)

    # Get distribution
    def SimFcnAD(indx):
        X = ADtmp.loc[indx,:].values
        dfit = distfit(n_boots=10)
        results = dfit.fit_transform(X)
        return dfit.model['model']
    
    # Get distribution
    def SimFcnNCI(indx):
        X = NCItmp.loc[indx,:].values
        dfit = distfit(n_boots=10)
        results = dfit.fit_transform(X)
        return dfit.model['model']

    # Get models
    ADmodels = Parallel(n_jobs=cpu_count())(delayed(SimFcnAD)(i) for i in range(len(ADtmp)))
    NCImodels = Parallel(n_jobs=cpu_count())(delayed(SimFcnNCI)(i) for i in range(len(NCItmp)))
    
    # get RV function for each model 
    def getRV(model):
        return model.rvs(size=args.runs)

    # joblib function that get RVs for all the genes
    def DataCentricSimulationAD():
        rvs = Parallel(n_jobs=cpu_count())(delayed(getRV)(m) for m in ADmodels)
        return rvs

    def DataCentricSimulationNCI():
        rvs = Parallel(n_jobs=cpu_count())(delayed(getRV)(m) for m in NCImodels)
        return rvs
    
    # run joblib function that get RVs for all the genes
    ADScores = DataCentricSimulationAD()
    NCIScores = DataCentricSimulationNCI()
   
    # Prep the data for scaling 
    ADScores = np.transpose(np.array(ADScores))
    NCIScores = np.transpose(np.array(NCIScores))
    unscaleddata = np.concatenate((ADScores,NCIScores))   
    unscaleddata = unscaleddata.clip(min = 0)

    # Set the scaler
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
    
    #Scale the data and split the data again to its original subsets
    scaledADdata = pd.DataFrame(np.transpose(scaleddata[:args.runs]))
    scaledADdata.to_csv('../{}/George-MC-sim-{}-{}-{}-{}-{}-{}-{}-AD-scores.csv'.format(args.folder,args.user,args.filter,args.runs,args.runno,args.scaler,args.column,args.columnmethod))
    
    for c in scaledADdata.columns:
        scaledADdata[c] = scaledADdata[c] * pd.Series(ADcoeff)
        
    scaledNCIdata = pd.DataFrame(np.transpose(scaleddata[args.runs:]))
    scaledNCIdata.to_csv('../{}/George-MC-sim-{}-{}-{}-{}-{}-{}-{}-NCI-scores.csv'.format(args.folder,args.user,args.filter,args.runs,args.runno,args.scaler,args.column,args.columnmethod))
    
    for c in scaledNCIdata.columns:
        scaledNCIdata[c] = scaledNCIdata[c] * pd.Series(NCIcoeff)
    
    # Get Scores 
    ADscores = np.exp(scaledADdata.sum())/(1+np.exp(scaledADdata.sum()))
    Ctrlscores = np.exp(scaledNCIdata.sum())/(1+np.exp(scaledNCIdata.sum()))
    
    #Save scores 
    print(ADscores)
    print(Ctrlscores)
    simruns = dict()
    simruns['AD Scores'] = ADscores
    simruns['NCI Scores'] = Ctrlscores
    simruns = pd.DataFrame(simruns)
    # Finding information about distribution of the results from both simulations
    simruns.to_csv('../{}/George-MC-sim-{}-{}-{}-{}-{}-{}-{}-simruns.csv'.format(args.folder,args.user,args.filter,args.runs,args.runno,args.scaler,args.column,args.columnmethod))

if __name__ == "__main__":
   main(sys.argv[1:])             
    
    
