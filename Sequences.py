import numpy as np
import pandas as pd

def create_sequences(df, target, selected_features, time_steps=10):
    """
    Create sequences of data for time series forecasting.

    Parameters:
    df (pd.DataFrame): The input dataframe containing features.
    target (pd.Series): The target variable.
    selected_features (list): List of feature names to use.
    time_steps (int, optional): The number of time steps for each sequence. Default is 10.

    Returns:
    np.array: Feature sequences.
    np.array: Corresponding target values.
    """
    X, Y = [], []
    for i in range(len(df) - time_steps):
        X.append(df[selected_features].iloc[i : i + time_steps].values)
        Y.append(target.iloc[i + time_steps])

    print("Features Data Shape:",np.array(X).shape)
    print("Target Data Shape:",np.array(Y).shape)
    return np.array(X), np.array(Y)