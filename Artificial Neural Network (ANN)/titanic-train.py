import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import seaborn as sns
import pickle #save encoder

# Data pre-processing
df_data = pd.read_csv('Titanic_data.csv')
df_names = pd.read_csv('Titanic_names.csv')

df = pd.merge(df_data, df_names, on='id', how='inner')

pclass = df['PClass'].value_counts()

# Remove PClass '*'
df = df[df.PClass != '*']

pclass = df['PClass'].value_counts()

df['Age']=df['Age'].replace(0, df['Age'][(df['Age']>0)].mean()) # fill 0 ages with mean age

# Divide X and y
X = df.loc[:, ['PClass', 'Age', 'Gender']]
y = df.loc[:, ['Survived']]


# Dummy variables
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['Gender','PClass'])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# Train and test data 80 % - 20 % 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Scale X
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Build and train ANN
model = Sequential()
model.add(Dense(8, input_dim=X.shape[1], kernel_initializer='normal', activation='relu')) # 8 size input layer
model.add(Dense(8, activation='relu')) # 8 size hidden layer
model.add(Dense(1, activation='sigmoid')) # 1 size output layer
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy','accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=8,  verbose=1, validation_data=(X_test,y_test))
    
# visualize training
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Predict with test data
y_pred = model.predict(X_test) # prosentteina
y_pred_class = y_pred > 0.5 # true / false

# Confusion Matrix and metrics
cm = confusion_matrix(y_test, y_pred_class)
acc = accuracy_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)

print(cm)
print (f'accuracy_score: {acc}')
print (f'recall_score: {recall}')
print (f'precision_score: {precision}')

print (f'y_test: {y_test.value_counts()}')

sns.heatmap(cm, annot=True, fmt='g')
plt.show()


# Save model to disk
model.save('titanic-model.h5')

# save encoder to disk
with open('titanic-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)
    
# save scalers to disk
with open('titanic-scaler_x.pickle', 'wb') as f:
    pickle.dump(scaler_x, f)
    


