# Import the required libraries
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz 
import os
# Set the path to the Graphviz executable
os.environ["PATH"] += os.pathsep +'s\Admin\Downloads\windows_10_msbuild_Release_graphviz-8.1.0-win32.zip ' # Replace with your Graphviz bin path
df=pd.read_csv('iris.csv')
df.head()
# Load the iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Create the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)
# Visualize the decision tree
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
display(graph)
