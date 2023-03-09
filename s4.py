from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearnex import patch_sklearn

# Load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Shuffle the data
X, y = shuffle(X, y, random_state=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Enable Intel Optimization for Scikit-Learn
patch_sklearn()

# Initialize the NuSVC classifier with Intel Optimization
clf = NuSVC(kernel='rbf', gamma='scale', random_state=0, max_iter=-1, decision_function_shape='ovr', probability=True, cache_size=1000)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the classes of the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
