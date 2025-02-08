

def user_select_features():
    """
    Prompt user to select features to use in the model
    """
    print("Select the features to include for training (comma-separated):")
    print("Available features: 'Open', 'High', 'Low', 'Volume', 'MA_10', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR', 'OBV'")
    selected_features = input("Enter the features: ").split(',')
    
    # Clean any extra spaces
    selected_features = [x.strip() for x in selected_features]
    print("Selected Features:",selected_features)
    return selected_features
