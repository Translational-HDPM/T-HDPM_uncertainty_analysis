#Monte Carlo simulation with linear combination type scoring function on Random data
#Author: George Rickus
import pandas as pd
import numpy as np
from timeit import default_timer as timer

def tester():
    start = timer()
    ### CSV File I created with random data points in excel(250 by 50)
    ### This is my file path in quotes. Replace with your own.
    myDF = pd.read_csv(".\OneDrive\Data_Mine\Summer_Work\Random_Data.csv")
    numVars = len(myDF.columns)
    numRuns = 100000
    
    # At each index "col" in means and stds, the value can be thought of as 
    # the mean and std of the normal distribution for that independent variable(EX. TPM distribution for Gene #col)
    means = [np.mean(myDF.loc[:, col]) for col in myDF.columns]
    stds = [np.std(myDF.loc[:, col]) for col in myDF.columns]
    
    # This function generates score from linear combination function based on mean and standard deviation for each variable
    coefficients = np.random.uniform(0, 0.5, numVars)
    
    # This function generates score from linear combination function based on mean and standard deviation for each variable
    def TermGen(means, stds, coefficients):
        score = 0
        for i in range(numVars):
            score += coefficients[i] * np.random.normal(means[i], stds[i])
        return score
    
    # Generating distribution of score values with "numRuns" scores
    scores = []
    for _ in range(numRuns):
        scores.append(TermGen(means, stds, coefficients))
    
    # Finding information about distribution
    outMean = np.mean(scores)
    outStd = np.std(scores)
    end = timer()
    return [end-start, outMean, outStd]

# The tester function itself is the entirety of the Monte Carlo simulation, but I thought it would be interesting to find
# the average time it took to run over "numTests" different trials based on my own local system specs and Purdue's computing cluster
times = []
finMeans = []
finStds = []
numTests = 10
for i in range(numTests):
    curr = tester()
    times.append(curr[0])
    finMeans.append(curr[1])
    finStds.append(curr[2])
print(np.mean(times))

