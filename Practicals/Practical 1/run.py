import pathlib

import pandas as pd

pathlib.Path('./output').mkdir(exist_ok=True)

# QUESTION 1

# read csv file for question 1
q1_file = pd.read_csv('./specs/AutoMpg_question1.csv')

# finding number of empty columns for horsepower and origin
empty_horsepower_columns = q1_file['horsepower'].isnull().sum()
empty_origin_columns = q1_file['origin'].isnull().sum()

# print the number of missing columns for horsepower and origin
print('Number of missing horsepower values: ', empty_horsepower_columns)
print('Number of missing origin columns', empty_origin_columns)

# find average of all horsepower values
horsepower_avg = q1_file['horsepower'].mean()
print('The average horsepower value is ', horsepower_avg)

# fill empty horsepower columns with average horsepower value
q1_file['horsepower'].fillna(q1_file['horsepower'].mean(), inplace=True)

# find minimum origin value
origin_min = q1_file['origin'].min()
print('Minimum origin is', origin_min)

#fill empty origin columns with minimum origin value
q1_file['origin'].fillna(q1_file['origin'].min(), inplace=True)

#save changes to file
q1_file.to_csv('./output/question1_out.csv', index=False)
print('Question 1 output saved')


# QUESTION 2

#read csv file for question 2
q2_file_a = pd.read_csv('./specs/AutoMpg_question2_a.csv')
q2_file_b = pd.read_csv('./specs/AutoMpg_question2_b.csv')

# rename column "name" in q2_file_b to "car name"
q2_file_b.rename(columns={'name': 'car name'}, inplace=True)
print('Column "name" renamed to "car name"')

# Create attribute "other" in A and give it default value 1
q2_file_a['other'] = 1
print('Column "other" created in B')

# Concatenate A and B
q2_file_c = pd.concat([q2_file_a, q2_file_b], sort=False)
print('A and B have been concatenated')

# Save result to output folder
q2_file_c.to_csv('./output/question2_out.csv', index=False)
print('Question 2 output saved')


