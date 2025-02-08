from data_preprocessing import preprocess_data, scale_data  
from user_select_features import user_select_features  
from FetchingData import fetch_Costum_data  
from Sequences import create_sequences  
from Evaluate import evaluate_model  
from LSTM_model import build_LSTM  
from Regression import Regression_model  

# Step 1: Fetch and preprocess data
df = fetch_Costum_data()  # Fetch custom data 
df = preprocess_data(df)  # Preprocess data (apply feature engineering)

# Step 2: User selects features for the model
selected_features = user_select_features()  # Ask user to select features for training the model

# Step 3: Validate the selected features
valid_features = ['Open', 'High', 'Low', 'Volume', 'MA_10', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR', 'OBV']  # List of valid features

# Filter selected features to make sure they are valid
selected_features = [f for f in selected_features if f in valid_features]
# If no valid features are selected, exit the program
if not selected_features:
    print("Invalid feature selection. Exiting.")
    exit()

# Step 4: Scale the data
# Scale the data using MinMaxScaler for both features and target variable
df_scaled, scaler_X, scaler_Y = scale_data(df, ['Close'], selected_features)

# Step 5: Prepare the data (extracting features and target)
X_data = df_scaled[selected_features].values  # Extract feature data
Y_data = df_scaled['Close']  # Extract target (stock closing price)

# Step 6: Select time steps for sequence creation
user_input = input("Enter the number of time steps (Press Enter to use default 10): ")
# Use the provided time steps or default to 10 if input is invalid
time_steps = int(user_input) if user_input.strip().isdigit() else 10
# Create sequences based on the time steps selected by the user
X, Y = create_sequences(df_scaled, Y_data, selected_features, time_steps)

# Step 7: Split the data into training and test sets (85% training, 15% testing)
split = int(0.85 * len(X))  # 85% for training data
X_train, X_test = X[:split], X[split:]  # Split features
Y_train, Y_test = Y[:split], Y[split:]  # Split target

# Step 8: User selects the model type to train
Model_Type = int(input("If you want to train and evaluate regression model press 0 or 1 for LSTM Model"))

# Step 9: Train the selected model
if Model_Type == 0:
    # For Regression model
    model = Regression_model()  # Initialize the regression model
    print("Regression model has been built. Now training.")
    # Train the regression model
    model.fit(X_train.reshape(X_train.shape[0], -1), Y_train)
elif Model_Type == 1:
    # For LSTM model
    model = build_LSTM(X_train.shape[1:])  # Initialize the LSTM model
    print("LSTM model has been built. Now training.")
    # Train the LSTM model
    model.fit(X_train, Y_train, batch_size=32, epochs=300)

# Step 10: Evaluate the model's performance
Results = evaluate_model(model, X_test, Y_test, scaler_Y, selected_features, Model_Type)
