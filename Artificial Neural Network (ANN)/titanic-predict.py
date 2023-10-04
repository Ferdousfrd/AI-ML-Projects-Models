import pandas as pd
from tensorflow.keras.models import load_model #load model
import pickle #load encoder

# load model
model = load_model('titanic-model.h5')

# load encoder
with open('titanic-ct.pickle', 'rb') as f:
    ct = pickle.load(f)
    
# load scalers
with open('titanic-scaler_x.pickle', 'rb') as f:
    scaler_x = pickle.load(f)


# predict with new data
Xnew = pd.read_csv('titanic-new.csv')
Xnew_org = Xnew
Xnew=ct.transform(Xnew)
Xnew= scaler_x.transform(Xnew)
ynew= model.predict(Xnew) 

# get scaled value back to unscaled
Xnew = scaler_x.inverse_transform(Xnew)

for i in range (len(ynew)):
    print (f'{Xnew_org.iloc[i]}\nStatus: {ynew[i][0]}\n')