import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Alpha Vantage API parameters
api_key = 'EPXDBQWPT3AE8AOL'
symbol = 'AAPL'
interval = 'daily'
output_size = 'full'

# Fetch stock data from Alpha Vantage API
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_{interval.upper()}_ADJUSTED&symbol={symbol}&outputsize={output_size}&apikey={api_key}'
response = requests.get(url)
data = response.json()

# Extract stock price data from the API response
time_series = data[f'Time Series ({interval.capitalize()})']
df = pd.DataFrame.from_dict(time_series, orient='index')
df.sort_index(ascending=True, inplace=True)
df = df[['1. open', '2. high', '3. low', '4. close', '5. adjusted close', '6. volume']]

# Convert '4. close' column to numeric data type
df['4. close'] = pd.to_numeric(df['4. close'])

# Calculate Simple Moving Average (SMA)
def calculate_sma(data, window):
    sma = data['4. close'].rolling(window=window).mean()
    return sma

# Calculate Relative Strength Index (RSI)
def calculate_rsi(data, window):
    delta = data['4. close'].diff()
    gains = delta.mask(delta < 0, 0)
    losses = -delta.mask(delta > 0, 0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate SMA and RSI features
df['SMA_20'] = calculate_sma(df, window=20)
df['SMA_50'] = calculate_sma(df, window=50)
df['RSI_14'] = calculate_rsi(df, window=14)

# Define the feature columns and target variable
feature_cols = ['SMA_20', 'SMA_50', 'RSI_14']
target_col = '4. close'

# Split the data into training and testing sets
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values in the feature columns
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make price predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Print the prediction for the next day
print("The predicted price for the next day is: ", y_pred[-1])
print ("-----------------------")

# Print the evaluation metrics
print ("Linear Regression Model")
print ("-----------------------")
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)