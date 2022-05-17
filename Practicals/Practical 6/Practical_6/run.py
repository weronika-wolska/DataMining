# -*- coding: utf-8 -*-
"""
Weronika Wolska - 17301623

COMP40370 - DATA MINING

Practical 6
"""

import pathlib
import pandas as pd

# make folder "Output"
pathlib.Path('./output').mkdir(exist_ok=True)

# QUESTION 1

# Part 1
#read file for q1
q1_file = pd.read_csv('./specs/question_1.csv')

# KMeans Algorithm
from sklearn.cluster import KMeans
import numpy as np

# define kmeans with nr. of clusters = 3, random_state=0, as per pdf
kmeans = KMeans(n_clusters=3, random_state=0)
#apply kmeans to q1_file
kmeans.fit(q1_file)
labels = kmeans.predict(q1_file)
centroids = kmeans.cluster_centers_
print("Kmeans algorithm applied.")

# Part 2 - add "cluster" column to csv file
q1_file['cluster'] = labels
q1_file.to_csv('./output/question_1.csv', index=False)
print("Q1 output saved.")


#Part 3
#make graph for q1
import matplotlib.pyplot as plt
colmap = {1: 'r', 2: 'g', 3: 'b'}
graph = plt.figure(figsize=(5,5))
colours = map(lambda x: colmap[x+1], labels)
colourss = list(colours)
plt.scatter(q1_file['x'], q1_file['y'], color=colourss, alpha=0.5)
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.show()
print("Q1 output graphed.")

# QUESTION 2

# read q2 file
q2_file = pd.read_csv('./specs/question_2.csv')


# Part 1 - Delete columns Name, Manuf, Type and Rating
remaining_cols = ['CALORIES', 'PROTEIN', 'FAT', 'SODIUM', 'FIBER', 'CARBO', 'SUGARS', 'POTASS', 'VITAMINS', 'SHELF', 'WEIGHT', 'CUPS']
q2_file = q2_file[remaining_cols]

# Part 2 - apply kmeans algorithm with k=10, 5 maximum runs, 100 maximum optimisation steps, random state 0
kmeans2 = KMeans(n_clusters=10, n_init=5, max_iter=100, random_state=0)
kmeans2.fit(q2_file)
labels2 = kmeans2.predict(q2_file)
centroids2 = kmeans2.cluster_centers_
print("kmeans algorith applied for question 2.2")
# save result to new column 'config1'
#q2_file['config1'] = labels2   Do this at the end, so that the column does not interfere with algorithms for the other parts of the question


# Part 3 - apply kmeans algorithm with k=10, 100 maximum runs, 100 maximum optimisation steps, random state 0
kmeans3 = KMeans(n_clusters=10, n_init=100, max_iter=100, random_state=0)
kmeans3.fit(q2_file)
labels3 = kmeans3.predict(q2_file)
centroids3 = kmeans3.cluster_centers_
print("kmeans algorithm applied for question 2.3")


# Part 5 - Run kmeans using k=3
kmeans5 = KMeans(n_clusters=3) 
kmeans5.fit(q2_file)
labels5 = kmeans5.predict(q2_file)
centroids5 = kmeans5.cluster_centers_
print("kmeans algorithm applied for question 2.5")

# Part 7 - save output

# add column 'config1'
q2_file['config1'] = labels2
q2_file['config2'] = labels3
q2_file['config3'] = labels5

# save to csv file
q2_file.to_csv('./output/question_2.csv')
print("output for question 2 saved.")

# QUESTION 3

# load question 3 file
q3_file = pd.read_csv('./specs/question_3.csv')

# Part 1 

# delete ID column
remaining_columns = ['x', 'y']
q3_file = q3_file[remaining_columns]

kmeans_q3 = KMeans(n_clusters=7, max_iter=100, n_init=5, random_state=0)
kmeans_q3.fit(q3_file)
labels_q3 = kmeans_q3.predict(q3_file)
centroids_q3 = kmeans_q3.cluster_centers_
print("kmeans algorithm applied to question 3.")
# save labels in new column 'kmeans'
q3_file['kmeans'] = labels_q3

# Part 2 - graph results
colmap_q3 = {1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm', 6: 'y', 7: 'k'}
graph_q3 = plt.figure(figsize=(5,5))
colours_q3 = map(lambda x: colmap_q3[x+1], labels_q3)
colourss_q3 = list(colours_q3)
plt.scatter(q3_file['x'], q3_file['y'], color=colourss_q3, alpha=0.5)
for idx, centroid in enumerate(centroids_q3):
    plt.scatter(*centroid, color=colmap_q3[idx+1])
plt.show()
print("Q3 kmeans graphed.")

# Part 3 

# Normalise x and y to be in range 0.0 - 1.0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(q3_file) 
scaled_q3_file = scaler.transform(q3_file)
print('normalised data: ', scaled_q3_file)

# perform DBSCAN clustering on normalised data for q3
from sklearn.cluster import DBSCAN
from collections import Counter
clustering = DBSCAN(eps=0.04, min_samples=4, metric='euclidean').fit(scaled_q3_file)
outliers = q3_file[clustering.labels_ == -1]
clusters = q3_file[clustering.labels_ != -1]
colours = clustering.labels_
colours_clusters = colours[colours != -1]
colours_outliers = 'black'

cl_nr = Counter(clustering.labels_)
print(cl_nr)
print('Number of clusters = {}'.format(len(clusters)-1))

plot = plt.figure()
pl = plot.add_axes([.1,.1,1,1])
pl.scatter(clusters['x'], clusters['y'], c=colours_clusters)
pl.scatter(outliers['x'], outliers['y'], c=colours_outliers)
plt.show()


clustering2 = DBSCAN(eps=0.08, min_samples=4, metric='euclidean').fit(scaled_q3_file)
outliers2 = q3_file[clustering2.labels_ == -1]
clusters2 = q3_file[clustering2.labels_ != -1]
colours2 = clustering2.labels_
colours_clusters = colours2[colours2 != -1]
colours_outliers = 'black'
cl_nr = Counter(clustering2.labels_)
print(cl_nr)
print('Number of clusters = {}'.format(len(clusters2)-1))

plot = plt.figure()
pl = plot.add_axes([.1,.1,1,1])
pl.scatter(clusters2['x'], clusters2['y'], c=colours_clusters)
pl.scatter(outliers2['x'], outliers2['y'], c=colours_outliers)
plt.show()

# add new columns to q3_file
q3_file['kmeans'] = labels_q3
q3_file['dbscan1'] = clustering.labels_
q3_file['dbscan2'] = clustering2.labels_
q3_file.to_csv('./output/question_3.csv')