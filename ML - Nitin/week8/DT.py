import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
# Reading the csv Iris dataset file
df = pd.read_csv("/home/machine/Desktop/ML/Iris.csv")
print(df.head(10))

# Conditioning the data
from sklearn.preprocessing import LabelEncoder

array = df.values
X = array[:,0:4]
y = array[:,4]
le = LabelEncoder()
y = le.fit_transform(y)

# Condition the training and Testing data
# The number of samples can be tuned with the test_size parameter.
# Here, 95% of the data is used.
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.40,random_state=0)
# Train the model using DecisionTreeClassifier ID3
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X, y)
pred = model.predict(X_test)
# Prepare confusion matrix
print("\n\nThe confusion matrix is \n\n")
conf = confusion_matrix(y_test,pred)
print(conf)
# pepare Classification Report
print("\n\nAccuracy is")
accuracy = accuracy_score(y_test,pred)
print(accuracy)
# Or report can be obtained as follows
print('\n Classification Report')
report = classification_report(y_test,pred)
print(report)
