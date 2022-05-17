# COMP40370 Data Mining
# Practical 2
# Weronika Wolska
# 17301623

import pathlib
import matplotlib.pyplot as plt
import pandas as pd

pathlib.Path('./output').mkdir(exist_ok=True)

# function to calculate z-score used for Q1.2
def zscore(val,avg, std_div):
    return (val - avg) / std_div

# function to calculate min/max transformation for Q1.3
def minmax(val, minval, maxval):
    return (val - minval) / (maxval - minval) 

# QUESTION 1: DATA TRANSFORMATION

# read CSV file for question 1
q1_file =  pd.read_csv('./specs/SensorData_question1.csv')

# Part 1

# Copy values from Input3 into Original Input3
q1_file['Original Input3']=q1_file['Input3']

# Copy values from Input12 into Original Input12
q1_file['Original Input12'] = q1_file['Input12']

# Part 2

# array for Input3 column
input3_array = q1_file['Input3']
print(input3_array)

avg = q1_file['Input3'].mean()
print('Mean average for Input3 is: ', avg)
std_div = q1_file['Input3'].std()
print('standard deviation for Input3 is: ', std_div)

#size = q1_file.length
#size = size.toInt(size)
for i in range(0,199):
    input3_array[i] = zscore(input3_array[i], avg, std_div)
    

q1_file['Input3'] = input3_array
    
# Part 3

input12 = q1_file['Input12']

maxval = q1_file['Input12'].max()
print('max value for Input12 is: ', maxval)
minval = q1_file['Input12'].min()
print('min value for Input12 is: ', minval)

for i in range(0,199):
    input12[i] = minmax(input12[i], minval, maxval)
    
q1_file['Input12'] = input12

print(q1_file['Input1'][0])

# Part 4
avg_column = [0]*199
print(len(avg_column))
print(len(q1_file['Input1']))


curr_row = [0]*12
for i in range(0,199):
    sum= q1_file['Input1'][i] + q1_file['Input2'][i] + q1_file['Input3'][i] + q1_file['Input4'][i] + q1_file['Input5'][i] + q1_file['Input6'][i] + q1_file['Input7'][i] + q1_file['Input8'][i] + q1_file['Input9'][i] + q1_file['Input10'][i] + q1_file['Input11'][i] + q1_file['Input12'][i]
    avg_column[i] = sum/12
    
q1_file['Average Input'] = avg_column
    
print('Average Input column created')
    
# Part 5

q1_file.to_csv('./output/question1_out.csv', index=False)
print('Question 1 output saved')

# QUESTION 2

# read input file for question 2
q2_file = pd.read_csv('./specs/DNAData_question2.csv')

# Part 1

#scale down all the values in the file to be between 0.0 and 1.0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(q2_file)
    
scaled_q2_file = scaler.transform(q2_file)

# import PCA from sklearn
from sklearn.decomposition import PCA

# reduce number of variants with at least 95% variance explained
pca = PCA(0.95)
pca.fit(scaled_q2_file)
print(scaled_q2_file)
#q2_output = pd.to_csv(scaled_q2_file)


# Part 2
from sklearn.preprocessing import KBinsDiscretizer
#from feature_engine.discretisers import EqualWidthDiscretiser

discretiser = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
discretiser.fit(scaled_q2_file)
#print(scaled_q2_file)

    


# Part 3


# Part 4 - save the genarated dataset

output_q2 = pd.DataFrame(scaled_q2_file)
print('length:', output_q2.size)

    
output_q2.to_csv('./output/question2_out.csv', index=False)
print('Question 2 output saved')
