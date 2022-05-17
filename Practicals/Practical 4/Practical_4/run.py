# Practical 4 code
# Weronika Wolska
# 17301623

# Question 1 - Simple Linear Regression

# Part 1

# load in data file and create output folder
import pathlib
import pandas as pd

pathlib.Path('./output').mkdir(exist_ok=True)

q1_file = pd.read_csv('./specs/marks_question1.csv')

# make graph
import matplotlib.pyplot as plt
plt.scatter(q1_file['midterm'], q1_file['final'])
plt.xlabel('midterm')
plt.ylabel('final')
plt.show()
#not working
#plt.savefig('./output/marks.png')

# Part 2

import numpy as np
from sklearn.linear_model import LinearRegression

# need to work with numpy.ndarray data type for this part
# reshape((-1,1)) ensures that the array is 2-dimensional
a = np.array(q1_file['midterm']).reshape((-1,1))
b = np.array(q1_file['final'])

# create model using default parameters
model = LinearRegression().fit(a, b)

# get prediction for final grades based on midterm grades
final_prediction = model.predict(a)
print(final_prediction)

#graph the prediction to the midterm
# this is for reference purposes for the report
plt.scatter(q1_file['midterm'], final_prediction)
plt.xlabel('midterm')
plt.ylabel('final prediction')
plt.show()


# QUESTION 2 - Classification with Decision Tree

#read file for question 2
q2_file = pd.read_csv('./specs/borrower_question2.csv')

# delete 'TID' column
to_keep = ['HomeOwner', 'MaritalStatus', 'AnnualIncome', 'DefaultedBorrower']
q2_file = q2_file[to_keep]

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# define X and y

required_input = ['HomeOwner', 'MaritalStatus', 'AnnualIncome']
#X
input_data = q2_file[required_input]

#y
target = q2_file['DefaultedBorrower']

tree = DecisionTreeClassifier(criterion ='entropy', min_impurity_decrease=0.5 )
tree.fit(input_data, target)
tree.predict(input_data)

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

data = StringIO()
export_graphviz(tree, out_file = data, filled=True, rounded = True, special_characters=True, feature_names=required_input, class_names=['No','Yes'])
graph = pydotplus.graph_from_dot_data(data.getvalue())
graph.write_png('./output/tree_high.png')
Image(graph.create_png())

tree2 = DecisionTreeClassifier(criterion ='entropy', min_impurity_decrease=0.1 )
tree2.fit(input_data, target)
tree2.predict(input_data)

data2 = StringIO()
export_graphviz(tree2, out_file = data2, filled=True, rounded = True, special_characters=True, feature_names=required_input, class_names=['No','Yes'])
graph2 = pydotplus.graph_from_dot_data(data2.getvalue())
graph2.write_png('./output/tree_low.png')
Image(graph2.create_png())


