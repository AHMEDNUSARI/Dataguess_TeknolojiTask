# Apple Stock Price Prediction Using Deep Learning and Regression Methods

![Stock Prediction](https://githubusercontent.com/AHMEDNUSARI/Dataguess_TeknolojiTask/main/StockImage.webp)
## 📌 Overview
This project was developed as part of an assessment task assigned by Dataguess Teknoloji. The objective is to build a machine learning model to predict Apple's stock price movements using historical financial data. This task evaluates technical expertise, problem-solving abilities, and familiarity with financial data by implementing various prediction techniques, including deep learning and regression methods.

The project leverages advanced machine learning algorithms to analyze past stock trends and generate forecasts, which can be valuable for investors and analysts.

📊 Applied Models
In this project, Linear Regression and LSTM with Attention Mechanism were applied separately to predict Apple's stock price between January 1, 2008, and February 1, 2025.

## 🔍 Why Use Both Models?
The goal of using these two distinct approaches is to compare traditional regression techniques with deep learning-based time series forecasting:

  * Linear Regression: A simple yet effective statistical model that captures linear relationships in historical stock data. It serves as a baseline model for comparison.
  * LSTM + Attention Mechanism: A deep learning model designed for sequential data, capable of capturing complex patterns, long-term dependencies, and non-linear trends in stock price movements. The attention mechanism enhances the model’s focus on the most influential past data points, improving accuracy.
  * 
By evaluating both models separately, this project analyzes their strengths and weaknesses, providing insights into which approach performs better for stock price prediction under different conditions. 🚀


## 📂 Repository Overview
This repository contains two types of training processes:

### Training with Apple Stock Price Data:
The models have been trained on Apple's stock price data between January 1, 2008, and February 1, 2025. This training process specifically focuses on predicting stock price movements using both Linear Regression and LSTM with Attention Mechanism.

### Reusable Model Training for Other Datasets:
The repository is designed to allow easy reuse of the models for training on different stock datasets. Users can retrain the models with data from Yahoo Finance stock market data source by following a simple process outlined in the repository. This flexibility makes it easy to apply the models to other stocks or financial data, providing a scalable solution for stock price prediction across different markets.



## 🖥️ How to Run the Code
As mentioned earlier, there are two training processes, and each has its own Python script. Below are the details on how to run each script:

### 1. Training with Apple Stock Price Data (Default Model)
  * Script Name: TrainDefultData.py
  * #### Steps to Run:
    1. Run the script using the following command:
       python TrainDefultData.py
    2. The script will automatically download Apple stock price data between January 1, 2008, and February 1, 2025.
    3. It will then perform data processing and select default features: ['Open', 'High', 'Low', 'Volume', 'MA_10', 'MACD', 'OBV'].
    4. The user will be prompted to enter 0 or 1 to select the model type:
       * 0: Linear Regression
       * 1: LSTM with Attention Mechanism
     5. After training the selected model, the results will be visualized and the following evaluation metrics will be displayed:
       * RMSE (Root Mean Square Error)
       * MAPE (Mean Absolute Percentage Error)
       * R² (Coefficient of Determination)



