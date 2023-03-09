from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Initialize the Random Forest Classifier with Intel Optimization
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, random_state=0, verbose=0)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the classes of the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
