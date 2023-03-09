import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearnex import patch_sklearn
patch_sklearn()

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use the Intel optimized version of SVM
svc = SVC(kernel='rbf', C=1.0, gamma='scale')

# Fit the SVM model to the training data
svc.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = svc.predict(X_test)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
