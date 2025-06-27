import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

if sys.version_info >= (3, 12):
    raise RuntimeError("Python 3.12+ not supported. Use Python 3.10")
    
start = '2004-01-01'
end = '2024-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker' , 'AAPL')
df = yf.download(user_input, start=start, end=end)
df.columns = df.columns.droplevel(1)


if df.empty:
    st.error("Invalid ticker or no data available. Please enter another ticker")
    st.stop()


#Describing Data
st.subheader('Data from 2004 - 2024')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


#Splitting data into training(70%) and testing(30%)
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#Scaling data between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load my model
model = load_model('keras_model_for_stock_trend_predicton.h5')


#Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
# as scaler will not be same for all predictions
scale_factor = 1/scaler[0]                     
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph
st.subheader('Predicted vs Original')
fig2 = plt.figure(figsize=(16,8))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


#Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import streamlit as st

# Calculate metrics
mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_predicted) / np.maximum(np.abs(y_test), 1e-8))) * 100
r2 = r2_score(y_test, y_predicted)

# Display metrics in formatted subheader
st.subheader(
    f"""Model Performance\n\n
    MAE: {mae:.2f}
    MSE: {mse:.2f}
    RMSE: {rmse:.2f}
    R²: {r2:.2f}"""
)












































































