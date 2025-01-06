# FPR-22098314
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Explicitly set matplotlib to use a different font
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set Seaborn style for better visuals
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# Load and preprocess data
df = pd.read_csv("/content/AAPL.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Display dataset summary
print("\nDataset Information:")
print(df.info())
print("\nFirst few rows of data:")
print(df.head())

# Exploratory Data Analysis (EDA)
print("\nPerforming EDA...")

# 1. Close price over time
plt.figure(figsize=(16, 8))
plt.plot(df['Close'], label='Close Price', color='#1f77b4', linewidth=2)
plt.title('Apple Stock Close Price History', fontsize=20, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Distribution of closing prices
plt.figure(figsize=(12, 6))
sns.histplot(df['Close'], kde=True, bins=50, color='purple', edgecolor='black')
plt.title('Distribution of Closing Prices', fontsize=20, fontweight='bold')
plt.xlabel('Close Price USD ($)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Daily percentage change
df['Daily Change (%)'] = df['Close'].pct_change() * 100
plt.figure(figsize=(16, 8))
sns.lineplot(data=df['Daily Change (%)'], color='orange', linewidth=2)
plt.title('Daily Percentage Change in Closing Price', fontsize=20, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Daily Change (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Moving averages
df['50-Day MA'] = df['Close'].rolling(window=50).mean()
df['200-Day MA'] = df['Close'].rolling(window=200).mean()
plt.figure(figsize=(16, 8))
plt.plot(df['Close'], label='Close Price', color='#1f77b4', alpha=0.5, linewidth=1)
plt.plot(df['50-Day MA'], label='50-Day Moving Average', color='#2ca02c', linewidth=2)
plt.plot(df['200-Day MA'], label='200-Day Moving Average', color='#d62728', linewidth=2)
plt.title('Close Price with Moving Averages', fontsize=20, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Volume traded over time
plt.figure(figsize=(16, 8))
plt.plot(df['Volume'], label='Volume', color='#7f7f7f', alpha=0.7, linewidth=1)
plt.title('Volume Traded Over Time', fontsize=20, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Volume', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Seasonality decomposition
decompose_result = seasonal_decompose(df['Close'], model='multiplicative', period=252)
decompose_result.plot()
plt.suptitle('Seasonality Decomposition', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. Autocorrelation plot
plt.figure(figsize=(16, 8))
autocorrelation_plot(df['Close'])
plt.title('Autocorrelation of Close Price', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()

# 8. Check stationarity using ADF Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print("\nADF Test Results:")
    print(f"Test Statistic: {result[0]:.2f}")
    print(f"P-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    print(f"Number of Observations Used: {result[3]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.2f}")
    if result[1] <= 0.05:
        print("\nData is stationary.")
    else:
        print("\nData is not stationary.")

adf_test(df['Close'])

# LSTM Model
print("\nBuilding and training LSTM model...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Prepare LSTM datasets
def create_lstm_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_lstm_dataset(train_data, time_step)
X_test, y_test = create_lstm_dataset(test_data, time_step)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

# Compile and train
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1)

# Predictions
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)
actual_test_data = scaler.inverse_transform(test_data[time_step:])

# Calculate residuals (Difference between actual and predicted)
lstm_residuals = actual_test_data - lstm_predictions

# Plotting the residuals correctly
plt.figure(figsize=(16, 8))
plt.plot(df.index[train_size + time_step:], lstm_residuals, color='#d62728', linewidth=2)
plt.title('LSTM Model Residuals', fontsize=20, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Prophet Model
print("\nBuilding Prophet model...")
prophet_df = df.reset_index()[['Date', 'Close']]
prophet_df.columns = ['ds', 'y']

# Define product launch dates for additional event handling
product_launch_dates = ['2020-09-15', '2021-09-14', '2022-09-07']
product_launch_dates = pd.to_datetime(product_launch_dates)

# Add product launch events to the data (as an additional regressor)
prophet_df['product_launch'] = prophet_df['ds'].apply(lambda x: 1 if x in product_launch_dates else 0)

# Build and fit Prophet model with product launch events
prophet_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True, 
    daily_seasonality=False
)
prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
prophet_model.add_country_holidays(country_name='US')

# Add the product launch event as a regressor
prophet_model.add_regressor('product_launch')

# Fit the model
prophet_model.fit(prophet_df)

# Forecast without a future dataframe, just predicting for the same period
forecast = prophet_model.predict(prophet_df)

# Visualize components
prophet_model.plot_components(forecast)
plt.suptitle('Prophet Model Components with Events', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualize Predictions
plt.figure(figsize=(16, 8))
plt.plot(prophet_df['ds'], prophet_df['y'], label='Actual Data', color='#1f77b4', linewidth=2)
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='#ff7f0e', linewidth=2)
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3, label='Uncertainty Interval')
plt.title('Prophet Model: Actual vs Predicted with Events', fontsize=20, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Evaluation for Prophet
mse_prophet = mean_squared_error(prophet_df['y'], forecast['yhat'])
print("\nProphet MSE:", mse_prophet)

# Evaluation for LSTM
mse_lstm = mean_squared_error(actual_test_data, lstm_predictions)
print("\nLSTM MSE:", mse_lstm)

# Residual Histogram for LSTM
plt.figure(figsize=(16, 8))
sns.histplot(lstm_residuals, bins=50, kde=True, color='#d62728')
plt.title('LSTM Model Residuals Distribution', fontsize=20, fontweight='bold')
plt.xlabel('Residuals', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Compare Prophet and LSTM Model MSE
print("\nComparison of Model Performance:")
print(f"Prophet MSE: {mse_prophet}")
print(f"LSTM MSE: {mse_lstm}")
