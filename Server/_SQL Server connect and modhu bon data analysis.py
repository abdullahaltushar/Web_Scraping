
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from numpy import array
import seaborn as sns

import pyodbc 
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-DK6PNJP;'
                      'Database=ASL_MCDL_LOCALTEST;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()


get_ipython().run_line_magic('matplotlib', 'inline')


SQL_Query = pd.read_sql_query('select * FROM STK_ITEM', conn)
data_location = pd.DataFrame(SQL_Query)
saledata = pd.read_excel("MODHU_DATA.xlsx", sheet_name='SALE')
list1=dict(zip(data_location.ITEMID,data_location.ITEMNM))
list1



data=saledata.ITEMID.map(list1)
data=pd.DataFrame(data)
data = pd.concat([saledata, data], axis=1)
data.columns.values[40] = "ItemName"


data=data[['QTY','TRANSDT','AMOUNT','RATE','ItemName']]
data.loc[data['QTY'] ==data['QTY'].max()]
data.loc[data['QTY'] ==data['QTY'].min()]
data.loc[data['AMOUNT'] ==data['AMOUNT'].max()]
data.loc[data['AMOUNT'] ==data['AMOUNT'].min()]
data.loc[data['RATE'] ==data['RATE'].max()]

data.loc[data['RATE'] ==data['RATE'].min()]
data.loc[data['ItemName'] =='PAHIM']

data1=data.loc[data['ItemName'] =='SP KACHA GOLLA']





data1=data1.groupby("TRANSDT").sum()
data1=data1.reset_index()


train_dates=pd.to_datetime(data1['TRANSDT'])
cols=list(data1)[1:4]

#df_for_training=data1[['QTY','AMOUNT','RATE']].astype(float)

print('Training set shape == {}'.format(train_dates.shape))
print('Featured selected: {}'.format(cols))

 # Data pre-processing
df_for_training=data[cols].astype(float)
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

trainX = []
trainY = []

n_future =1
n_past = 3
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

model = Sequential()
model.add(LSTM(units=50,activation='relu', return_sequences=True,  input_shape=(trainX.shape[1], trainX.shape[2])))

model.add(LSTM(units=50,activation='relu',return_sequences=True))

model.add(LSTM(units=50,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

#start Training


history = model.fit(trainX, trainY, epochs=10,batch_size=64,validation_split=0.1, verbose=1)
n_days_for_prediction=28
#n_past = 14

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='1d').tolist()
prediction = model.predict(trainX[-n_days_for_prediction:]) 


prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]


# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'TRANSDT':np.array(forecast_dates), 'QTY':y_pred_future})
df_forecast['TRANSDT']=pd.to_datetime(df_forecast['TRANSDT'])


original = data1[['TRANSDT', 'QTY']]
original['TRANSDT']=pd.to_datetime(original['TRANSDT'])
#original = original.loc[original['TRANSDT'] >= '2022-01-01']

sns.lineplot(original['TRANSDT'], original['QTY'])
sns.lineplot(df_forecast['TRANSDT'], df_forecast['QTY'])





