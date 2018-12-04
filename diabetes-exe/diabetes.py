import numpy as np
from pandas import read_csv
from sklearn import preprocessing

dataset = read_csv('diabetes.csv')
print(dataset.describe())


######  mark missing data

print((dataset[['blood pressure']] == 0).sum())

#replace missing data with NaN
dataset['blood pressure'] = dataset['blood pressure'].replace(0, np.NaN)
???

print(dataset.describe())


#ToDo:
#normilize values
# use Imputer to deal with missing values
# use OneHotEncode for categorial data
# balance the data
# fill missing values with mean column values


# fill missing values with mean column values
names=dataset.columns._data[:-1]

X=dataset.iloc[:,:9];
y=dataset.iloc[:,9];

??????


from sklearn import tree
#TODO: define a decision tree classifier with max depth 5.
# train and print the score
clf = tree.DecisionTreeClassifier(???)



# creating a print of the graph
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=names, filled=True, rounded=True, class_names=['sick','healthy'] )
graph = graphviz.Source(dot_data)
graph.render("diabetes_tree")