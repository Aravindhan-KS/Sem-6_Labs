from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Train the model using DecisionTreeClassifier ID3
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model = clf.fit(X, y)
#Visualize the model using tree graph
fig = plt.figure(figsize=(10,8))
_ = tree.plot_tree(clf, 
 feature_names=iris.feature_names,
 class_names=iris.target_names,
 filled=True)
plt.show()