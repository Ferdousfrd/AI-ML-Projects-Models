import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


#importing dataset
companies = pd.read_csv('D:/ai-ml/linReg/1000_Companies.csv')
X = companies.iloc[:,: -1].values
y = companies.iloc[:, 4].values

#showing the data
companies.head()

#data visualisation
#building the correlation maatrix
sns.heatmap(companies.corr())

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:, 3])

# Create a ColumnTransformer with OneHotEncoder for specific columns
from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(), [3])  # Replace [3] with the index of the column you want to one-hot encode
        # You can add more transformers for other columns if needed
    ],
    remainder="passthrough"  # Pass through columns that are not specified for one-hot encoding
)

# Fit and transform your data using the ColumnTransformer
X_transformed = column_transformer.fit_transform(X)

X = X[:, 1:]



####

#splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction the test set results
y_pred = regressor.predict(X_test)

#calculating the R squared value
r2_score(y_test, y_pred)
print(r2_score(y_test, y_pred))