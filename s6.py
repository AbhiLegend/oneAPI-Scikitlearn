from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
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

# Initialize the ElasticNet model with Intel Optimization
enet = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=0, selection='cyclic')

# Fit the model to the training data
enet.fit(X_train, y_train)

# Predict the target values of the testing data
y_pred = enet.predict(X_test)

# Calculate the R2 score of the model
r2 = r2_score(y_test, y_pred)

print('R2 score:', r2)
