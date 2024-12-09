#My first Monte Carlo simulation attempt: Dice Roll
#Author: George Rickus
import numpy as np

#Function that Generates artificial distribution with n data points(dice rolls).
def DataGen(n: int):
    #Independent variable sampling function: EX. np.random.random, np.random.uniform
    artificial_data = 6 * np.random.uniform(0, 1, n)
    total = [0, 0, 0, 0, 0, 0]
    for i in range(len(artificial_data)):
        total[int(artificial_data[i])]+=1
    for j in range(6):
        total[j] = total[j]/n
    return total

#Creating distribution
dist = DataGen(10000)
#Possible outcomes
options = [i for i in range(6)]
#Number of times random sampling is taken based on distribution
total_runs = 1000000
#Output data generation
totals = [0 for i in range(6)]
for i in range(total_runs):
    totals[np.random.choice(options, p=dist)] += 1
#Finding distribution and related information
outDist = [count/total_runs for count in totals]
mean = np.mean(outDist, axis=0)
std = np.std(outDist, axis=0)
print(outDist)
print(totals)
print(mean)
print(std)


