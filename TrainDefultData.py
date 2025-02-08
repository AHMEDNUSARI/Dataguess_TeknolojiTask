from data_preprocessing import preprocess_data, scale_data  
from FetchingData import fetch_data  
from Sequences import create_sequences  
from Evaluate import evaluate_model  
from LSTM_model import build_LSTM  
from Regression import Regression_model 

 

# Step 1: Fetch and preprocess data
df = fetch_data()  # Fetch stock data (for example, Apple stock data)
df = preprocess_data(df)  # Preprocess data (apply feature engineering)

# Step 2: User input to select model type
Model_Type = int(input("If you want to train and evaluate regression model press 0 or 1 for LSTM Model"))

# Step 3: Select features for training the model
selected_features = ['Open', 'High', 'Low', 'Volume', 'MA_10', 'MACD', 'OBV']  # The features used for model training

# Step 4: Check model type selection
if Model_Type == 0:
    # For Regression Model
    # Scale the data (for both features and target)
    df_scaled, scaler_X, scaler_Y = scale_data(df, ['Close'], selected_features)  
    X_data = df_scaled[selected_features].values  # Extract features
    Y_data = df_scaled['Close']  # Extract target (stock closing price)
    
    print("As a default, time steps is 10")
    # Create sequences for time series data (based on time steps)
    X, Y = create_sequences(df_scaled, Y_data, selected_features, time_steps=10)
    
    # Split the data into training (85%) and testing (15%) sets
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    # Initialize the regression model
    model = Regression_model()
    print("Regression model has been built. Now training.")
    # Train the regression model using the training data
    model.fit(X_train.reshape(X_train.shape[0], -1), Y_train)
    
elif Model_Type == 1:
    # For LSTM Model
    # Scale the data (for both features and target)
    df_scaled, scaler_X, scaler_Y = scale_data(df, ['Close'], selected_features)
    X_data = df_scaled[selected_features].values  # Extract features
    Y_data = df_scaled['Close']  # Extract target (stock closing price)
    
    print("As a default, time steps is 10")
    # Create sequences for time series data (based on time steps)
    X, Y = create_sequences(df_scaled, Y_data, selected_features, time_steps=10)
    
    # Split the data into training (85%) and testing (15%) sets
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    
    # Initialize the LSTM model
    model = build_LSTM(X_train.shape[1:])
    print("LSTM model has been built. Now training.")
    # Train the LSTM model using the training data
    model.fit(X_train, Y_train, batch_size=32, epochs=300)

# Step 5: Evaluate the model's performance
Results = evaluate_model(model, X_test, Y_test, scaler_Y, selected_features, Model_Type)
