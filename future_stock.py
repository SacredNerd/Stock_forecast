import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import os
import pandas_datareader as dat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential 
from keras.layers import Dense, LSTM
#import tensorflow as tf


st.title('STOCK FORECASTING')

# Gathering data
tick = st.text_input('Enter Stock Ticker', 'AAPL') 
data = yf.download(tickers=tick,period='5y',interval='1d')

# Describing data
subheader = 'Data from last 5 years: '
st.subheader(subheader)
st.write(data.describe())

clos = data.reset_index()['Close'] # market open data
ds = clos.values # converting into numpy array

# Viualizing market Close data
st.write('\n\n')
st.subheader('Visualizing Market Close Data')
fig = plt.figure(figsize = (20,10)) 
plt.plot(clos)
st.pyplot(fig)

#normalizing data between 0 and 1 (using MinMaxScaler)
normalizer = MinMaxScaler(feature_range=(0, 1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

#defining training and testing data frames
train_size = int(len(ds_scaled)*0.65)
test_size = len(ds_scaled) - train_size

#splitting data between train and test
ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

def create_ds(dataset,step=1):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i+step, 0])
    return np.array(Xtrain), np.array(Ytrain)

#creating the dataset for training
time_stamp = 100
x_train, y_train = create_ds(ds_train, time_stamp)
x_test, y_test = create_ds(ds_test, time_stamp)


#reshaping data to fit into a LSTM model
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#creating/loading the model
fil = str(tick)+'.h5'
if os.path.exists(fil):
    Model = load_model(fil)
else:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences = True, input_shape = (x_train.shape[1], 1))) 
    model.add(LSTM(units=50, return_sequences = True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    # model.compile(optimizer='sgd',loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=64,verbose=1)
    
    fil = str(tick)+'.h5'
    model.save(fil)
    Model = load_model(fil)

#predicting on train and test data
train_predict = Model.predict(x_train)
test_predict = Model.predict(x_test)

#inverse transform to get actual value
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

#comparing using visuals
fig = plt.figure(figsize = (20,10)) 
look_back = 100

#shift train predictions for plotting
trainPredictPlot = np.empty_like(ds_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

#shift test predictions for plotting
testPredictPlot = np.empty_like(ds_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(ds_scaled)-1,:] = test_predict

#plot the graph
st.write('\n\n')
st.subheader('Training and Testing the Model using LSTM ')
plt.plot(normalizer.inverse_transform(ds_scaled), label='Original data')
plt.plot(trainPredictPlot, label='predicted train data')
plt.plot(testPredictPlot, label='predicted test data')
plt.legend()
st.pyplot(fig)

#future prediction begins here ...

fut_inp = ds_test[len(ds_test)-100:].reshape(1,-1)
tmp_inp = list(fut_inp)
tmp_inp = tmp_inp[0].tolist()

#Predicting next 30 days price using the current data

lst_output = []
n_steps=100
i=0

while(i<30):
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp = fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1)) 
        yhat = Model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1)) 
        yhat = Model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1

#Creating a dummy plane to plot graph one after another 
plot_new = np.arange(1,101)
plot_pred = np.arange(101,131)
st.write('\n\n')
st.subheader('Predicting next 30 days ...')
fig2 = plt.figure(figsize=(20,10))
plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[len(ds_scaled)-100:])) 
plt.plot(plot_pred, normalizer.inverse_transform(lst_output),label='predicted {0} days'.format(30))
plt.legend()
st.pyplot(fig2)

#combined graph
ds_complete = ds_scaled.tolist()
ds_complete.extend(lst_output)

#final graph
st.write('\n\n')
st.title('Final Graph')
fig2 = plt.figure(figsize=(20,10))
final_graph = normalizer.inverse_transform(ds_complete).tolist()

plt.plot(final_graph)
plt.ylabel("price")
plt.xlabel("Time")
plt.title(tick+' prediction of next month close ')
plt.axhline(y=final_graph[len(final_graph)-1], color='red',linestyle=':',label='NEXT 30D: {}'.format(round(float(str(*final_graph[-1])))))
plt.legend()
st.pyplot(fig2)