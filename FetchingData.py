import yfinance as yf


def fetch_data():

    '''
    If the user doesn't enter any of these, default values will be used:
    - Default ticker: AAPL
    - Default start date: 2008-01-01
    - Default end date: 2025-01-01

    It then retrieves the stock's historical data for the specified period and renames the columns for consistency.

    Returns:
        pandas.DataFrame: A DataFrame containing the stock's historical data with the following columns:
            - 'Close': Closing price
            - 'High': Highest price of the day
            - 'Low': Lowest price of the day
            - 'Open': Opening price
            - 'Volume': Trading volume
    '''


    default_ticker = "AAPL"
    default_start_date = "2008-01-01"
    default_end_date = "2025-02-01"
    
    df = yf.download(default_ticker, start=default_start_date, end=default_end_date)
    
    # Rename columns for consistency
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    print("First Ten Raw Data")
    print(df.head(10))
    
    return df

def fetch_Costum_data():
    """
    Fetches historical stock data from Yahoo Finance based on user input or default values.

    This function prompts the user to enter:
    - A stock ticker symbol (e.g., AAPL, TSLA, MSFT, AMZN, GOOGL)
    - A start date (YYYY-MM-DD) 
    - An end date (YYYY-MM-DD) 


    It then retrieves the stock's historical data for the specified period and renames the columns for consistency.

    Returns:
        pandas.DataFrame: A DataFrame containing the stock's historical data with the following columns:
            - 'Close': Closing price
            - 'High': Highest price of the day
            - 'Low': Lowest price of the day
            - 'Open': Opening price
            - 'Volume': Trading volume
    """
    
    ticker = input("Enter the stock ticker symbol (e.g., AAPL, TSLA, MSFT, AMZN, GOOGL): ").strip().upper()
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD) : ").strip()
    
    # Fetch data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Rename columns for consistency
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    print("First Ten Raw Data")
    print(df.head(10))
    return df