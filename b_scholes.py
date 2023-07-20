import requests
from math import log, sqrt, exp

# Function to calculate the Black-Scholes option price
def calculate_option_price(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    d1 = (log(stock_price / strike_price) + (risk_free_rate + (volatility ** 2) / 2) * time_to_maturity) / (volatility * sqrt(time_to_maturity))
    d2 = d1 - volatility * sqrt(time_to_maturity)
    
    # Assuming a call option
    option_price = stock_price * cumulative_distribution(d1) - strike_price * exp(-risk_free_rate * time_to_maturity) * cumulative_distribution(d2)
    return option_price

# Function to calculate the cumulative distribution function
def cumulative_distribution(x):
    # Approximation for the cumulative distribution function (CDF)
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    L = abs(x)
    k = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / sqrt(2 * 3.141592653589793) * exp(-L * L / 2.0) * (a1 * k + a2 * k * k + a3 * k * k * k + a4 * k * k * k * k + a5 * k * k * k * k * k)
    if x < 0:
        w = 1.0 - w
    return w

# Main function
def main():
    # API endpoint and your Alpha Vantage API key
    api_endpoint = "https://www.alphavantage.co/query"
    api_key = "EPXDBQWPT3AE8AOL"
    
    # Get the stock ticker symbol from user input
    symbol = input("Enter the stock ticker symbol: ")
    
    # Make the API call to retrieve stock prices
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key
    }
    
    try:
        response = requests.get(api_endpoint, params=params)
        data = response.json()
        # Assuming the API response contains stock prices in "Time Series (Daily)"
        time_series = data["Time Series (Daily)"]
        latest_date = list(time_series.keys())[0]
        stock_price = float(time_series[latest_date]["4. close"])
        
        # Set the necessary inputs for the Black-Scholes model
        strike_price = float(input("Enter the option's strike price: "))
        time_to_maturity = float(input("Enter the time to option maturity in years: "))
        risk_free_rate = float(input("Enter the risk-free interest rate: "))
        volatility = float(input("Enter the annualized volatility: "))
        
        # Calculate the option price using the Black-Scholes model
        option_price = calculate_option_price(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility)
        
        # Print the option price on the screen
        print("Option Price: ${:.2f}".format(option_price))
        
    except KeyError:
        print("Error: Invalid stock ticker symbol or unable to fetch data.")
    
    except Exception as e:
        print("Error:", str(e))

# Execute the main function
if __name__ == "__main__":
    main()
