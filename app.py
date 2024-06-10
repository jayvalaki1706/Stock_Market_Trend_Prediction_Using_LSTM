import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import pandas_datareader.data as web
import datetime
from datetime import datetime
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Trend Prediction')

start = datetime(2010, 1, 1)
end = datetime(2019, 12, 31)
default_date = datetime(2015, 1, 1)


user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start=start, end=end)


# Describing Data
st.subheader('Data From 2010 - 2019')
st.write(df.describe())

# Visualization of Data
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100  = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100  = df.Close.rolling(100).mean()
ma200  = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

# splitting data into train and test
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing  = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


# load model
model = load_model('keras_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    

x_test,y_test = np.array(x_test),np.array(y_test)


y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


selected_date = st.date_input("Select a date between 2010 and 2019",value=default_date, min_value=start, max_value=end)
st.write(f"You selected: {selected_date}")
# Extract the row corresponding to the selected date
selected_date_str = selected_date.strftime('%Y-%m-%d')
if selected_date_str in df.index:
    selected_date_data = df.loc[selected_date_str]
    st.subheader(f'Stock Data for {selected_date_str}')
    st.write(selected_date_data)
else:
    st.write(f"No data available for {selected_date_str}")


