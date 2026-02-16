import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# Reading the csv Iris dataset file
df = pd.read_csv("/home/machine/Desktop/ML/Iris.csv")
print(df.head(10))
# Conditioning the data
array = df.values
X = array[:,0:4]
y = array[:,4]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) 

# Condition the training and Testing data
# The number of samples can be tuned with the test_size parameter.
# Here, 95% of the data is used.
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=0)
# Construct the Linear model
model = SVC(kernel='linear',random_state=0)
model.fit(X_train,y_train)
# Test the model with Linear kernel
pred = model.predict(X_test)
# Prepare confusion matrix
print("\n\nThe confusion matrix is \n\n")
conf = confusion_matrix(y_test,pred)
print(conf)
# pepare Classification Report
print("\n\nAccuracy is")
accuracy = accuracy_score(y_test,pred)
print(accuracy)
#RBF
model1 = SVC(kernel='rbf',random_state=0)
model1.fit(X_train,y_train)
# Test the model
pred = model1.predict(X_test)
# Prepare confusion matrix
print("\n\nThe confusion matrix for RBF kernel is \n\n")
conf = confusion_matrix(y_test,pred)
print(conf)
# pepare Classification Report
print("\n\nAccuracy for RBF kernel is")
accuracy1 = accuracy_score(y_test,pred)
print(accuracy1)
model2 = SVC(kernel='poly',random_state=0)
model2.fit(X_train,y_train)
# Test the model with Linear kernel
pred = model2.predict(X_test)
# Prepare confusion matrix
print("\n\nThe confusion matrix is \n\n")
conf = confusion_matrix(y_test,pred)
print(conf)
# pepare Classification Report
print("\n\nAccuracy is")
accuracy2 = accuracy_score(y_test,pred)
print(accuracy2)

print("LINEAR :",accuracy, "RBF :",accuracy1 ,"POLY :" ,accuracy2)