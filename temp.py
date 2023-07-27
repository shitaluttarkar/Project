# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Basic Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# imporing datareader to download the data from a website
import yfinance as yf
import datetime as dt
import streamlit as st
import pytz

# deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import load_model

# to remove warnings from the code
from warnings import filterwarnings
filterwarnings("ignore")

# streamlit
st.title("Apple Stock price prediction")
st.write("This is an application devoloped to predict the future APPLE Stock Price")


# use when buiding web app
user_input = st.text_input("Stock ticker", 'AAPL')
st.sidebar.header('User Input Parameters')
start = st.sidebar.date_input('Start Date', pd.to_datetime("2012-01-01"))
end   = st.sidebar.date_input('End Date Date', pd.to_datetime('today'))

df  = yf.download(user_input, start, end, ignore_tz=True)
st.subheader('Apple stock price Data')
st.write(df)

# describing data
#st.subheader("Description of data")
#st.write(df.describe())

# # visualization
st.subheader("Apple Close price chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label = 'Close price data')
#plt.xlabel("Date")
#plt.ylabel("Close Price")
plt.legend(loc='best')
st.pyplot(fig)

# # 125 and 250 days moving avarage
st.subheader("Apple Close price chart with 100 and 200 days moving avarage")
st.text('To analyse the Trend, it is common practice to use Moving Avarages.')
st.text('If 100 days moving avarage is greater then 200 days moving avarage, then it shows an UPTREND!')
st.text('If 100 days moving avarage is less then 200 days moving avarage, then it is shows DOWNTREND!')
ma_125 = df.Close.rolling(100).mean()
ma_250 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label = 'Orifinal Close price')
plt.plot(ma_125,label='100 days Moving Avarage')
plt.plot(ma_250,label='200 days Moving Avarage')
#plt.xlabel("Date")
#plt.ylabel("Close price")
plt.legend(loc='best')
st.pyplot(fig)

# splitting data into training and testing
train_data       = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
#validation_data  = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df)*0.80)])
test_data        = pd.DataFrame(df['Close'][int(len(df)*0.70):len(df)])

# converting data into scaled data(?)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) #to avoid 0 for lowest value of price
train_data_array = scaler.fit_transform(train_data)

# Data preparation for training x_train, y_train and creating lag data as features for the traing
x_train = []
y_train = []
n_features_of_lags = 100 
#No_of_lag_columns, this is a parameter given by user , n_features_of_lags=100 means value on the present day depends on past 100 days
for i in range(n_features_of_lags, len(train_data)):
    x_train.append(train_data_array[i-100:i,0])
    y_train.append(train_data_array[i,0])

#convering x_train and y_train in numpy array
x_train, y_train = np.array(x_train), np.array(y_train) 
# reshaping the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))


#load model
model =load_model('Trained_model.h5')

st.subheader("Trained vs Predicted prices")
# ploting results
fig3 = plt.figure(figsize=(12,5))
plt.plot(train_data['Close'][100:].values, label='original data')
plt.plot(np.reshape(scaler.inverse_transform(model.predict(x_train)), y_train.shape), label = 'predictions')
plt.xlabel("time step")
plt.legend(loc='best')
st.pyplot(fig3)


#Testing part
# we need to have 100 lag features to predict the close price in test so we need to append those values in test_data
past_100_days = train_data.tail(100)
new_test_data_with_index = past_100_days.append(test_data, ignore_index=False)
new_test_data = past_100_days.append(test_data, ignore_index=True)

# converting data into scaled data(?)
from sklearn.preprocessing import MinMaxScaler
scaler_test = MinMaxScaler()
input_data = scaler_test.fit_transform(new_test_data)

# Data preparation for testing
x_test = []
n_features_of_lags = 100
for i in range(n_features_of_lags, input_data.shape[0]):
    x_test.append(input_data[i-100: i, 0])

y_test = test_data['Close'].values

# converting into numpy array
x_test = np.array(x_test)

# reshaping the test data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

# making predictions
y_pred = scaler_test.inverse_transform(model.predict(x_test))
y_pred = np.reshape(y_pred, y_test.shape)

# Ploting prediction Vs true values
st.subheader("predictions vs true values for test data")
fig4 = plt.figure(figsize=(12,5))
plt.plot(y_test, label = 'Original price')
plt.plot(y_pred, label = 'Predicted price')
plt.xlabel("Time")
plt.ylabel("Close price")
plt.legend(loc='best')
st.pyplot(fig4)


# Next 30 days prediction
next_data = pd.DataFrame()
next_data['Close'] = []

n_days = st.sidebar.slider("No of days to Forcast", 1, 100)
# remember need to add next 30 days dates
# we need to have 100 lag features to predict the close price in next_data so we need to append those values in nextt_data
past_130_days_next = test_data.tail(n_features_of_lags+n_days)
# new_test_data_with_index = past_100_days.append(test_data, ignore_index=False)
new_next_data = past_130_days_next.append(next_data['Close'], ignore_index=True)
new_next_data = new_next_data[:-1]
# appliing minmax scaler
scaler_next = MinMaxScaler()
input_data_next = scaler_next.fit_transform(new_next_data)
# Data preparation for testing
x_next = []
for i in range(n_features_of_lags, input_data_next.shape[0]):
  x_next.append(input_data_next[i-100: i, 0])

# converting into numpy array
x_next = np.array(x_next)
# reshaping the test data
x_next = np.reshape(x_next, (x_next.shape[0], x_next.shape[1],1))
# making predictions for next 30 days
y_next = scaler_next.inverse_transform(model.predict(x_next))
y_next = np.reshape(y_next, (n_days,))
next_30 = pd.DataFrame()
next_30[' predicted Close price'] = y_next
#next_30.head()

# creating a function for weekdays
import holidays
Holiday = holidays.DE()
def getNextBusinessDay(date, n):
  for i in range(n):
    nextday = date + dt.timedelta(days=1) 
    while nextday.weekday()>4 or nextday in Holiday:
      nextday += dt.timedelta(days=1)
    date = nextday
  return date
  
# date = end date(today), n= no.of days price to be predicted
date = st.date_input('Today Date', pd.to_datetime('today'))
dates = []
for i in range(0,n_days):
  dates.append(getNextBusinessDay(dt.datetime.strptime(str(date),'%Y-%m-%d'),i))
#dates
next_30['Date'] = dates
# these range of needed to give from the external user!
# making datetime index to acess data using date
next_30.index = pd.to_datetime(next_30['Date']) 
next_30 =next_30.drop('Date', axis=1)
st.subheader("Forecasting Results")
st.write(next_30)
