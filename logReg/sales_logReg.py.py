import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns


### Prepare data ###

# Read data to pandas dataframe
df = pd.read_csv('adds_sales_data.csv')

# Remove column 'User ID'
df.pop('User ID')

# Convert Female/Male values into 1/0 values
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})


# Separate data into input data (X) and output data (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1].values


# Divide the data into the train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale (normalise) the values in columns Age and EstimatedSalary in range [0, 1]
# This scaling prevents the domination of columns with large values
age_max = X_train['Age'].max()
salary_max = X_train['EstimatedSalary'].max()
X_train['Age'] = X_train['Age'] / age_max
X_test['Age'] = X_test['Age'] / age_max
X_train['EstimatedSalary'] = X_train['EstimatedSalary'] / salary_max
X_test['EstimatedSalary'] = X_test['EstimatedSalary'] / salary_max


### Train the Logistic Regression model ###

#
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

####
#
# Now we have the model trained. Next the model will be used and tested with new data  
#
####

###### Test and evaluate the model with the test data (X_test, y_test), which was not used in training ######

# Predicting outputs with X_test as inputs
y_pred = model.predict(X_test)

### Create dataframe with input values (X_test), predicted outputs (y_pred) and real outputs (y_test) ###

# Create empty dataframe for the results
test_results = pd.DataFrame()

# Get Age and EstimatedSalary values from X_test
# Note! Because the values were scaled, the values must be scaled back to original scale (inverse_transform)
test_results = X_test

# Add the predicted outputs column
test_results['Predicted output'] = y_pred

# Add the real values from y_test
test_results['Real output'] = y_test

# Estimate the result by confusion matrix
# TP FN
# FP TN
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cbar=False)
plt.show()

# Check the accuracy of the results
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)
# Single input prediction, using the same scaling as it was used in training
d = {'Gender': [1], 'Age': [45 / age_max], 'EstimatedSalary': [88000 / salary_max]}
y_result = model.predict(pd.DataFrame(data=d))
