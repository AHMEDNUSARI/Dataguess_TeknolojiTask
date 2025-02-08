import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def evaluate_model(model, X_test, Y_test, scaler_Y, selected_features, Model_Type):
    """
    Evaluate the model performance and visualize the results.
    
    Parameters:
    model (keras.Model): Trained LSTM model.
    X_test (np.array): Test feature set.
    Y_test (np.array): Test target values.
    scaler_Y (sklearn.preprocessing.MinMaxScaler): Scaler used for target variable.
    selected_features (list): List of selected feature names.
    
    Returns:
    dict: A dictionary containing evaluation metrics (RMSE, MAPE, R²)
    """
    # Make predictions
    if Model_Type == 0:
        Y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    elif Model_Type == 1:
        Y_pred = model.predict(X_test)
    
    # Rescale predictions and test data
    Y_pred_rescaled = scaler_Y.inverse_transform(Y_pred.reshape(-1, 1))
    Y_test_rescaled = scaler_Y.inverse_transform(Y_test.reshape(-1, 1))
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(Y_test_rescaled, Y_pred_rescaled))
    mape = mean_absolute_percentage_error(Y_test_rescaled, Y_pred_rescaled)
    r2 = r2_score(Y_test_rescaled, Y_pred_rescaled)
    
    # Print metrics
    metrics = {'Selected Features': selected_features, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}
    print(metrics)
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test_rescaled, label='Actual Prices', color='blue')
    plt.plot(Y_pred_rescaled, label='Predicted Prices', color='red', linestyle='-')
    plt.title('Predicted vs Actual Prices', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.show()
    
    return metrics
