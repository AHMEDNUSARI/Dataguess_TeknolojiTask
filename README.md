# Apple Stock Price Prediction Using Deep Learning and Regression Methods
https://github.com/AHMEDNUSARI/Dataguess_TeknolojiTask/blob/main/Stock.png
![(https://github.com/AHMEDNUSARI/Dataguess_TeknolojiTask/blob/main/Stock.png)](https://github.com/AHMEDNUSARI/Dataguess_TeknolojiTask/blob/main/Stock.png)
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

## 📚 Project Reports
This project consists of three main reports that help in understanding the entire process of stock price prediction, from data analysis to model evaluation. These reports provide detailed explanations and visualizations to ensure a clear understanding of how each step contributes to the overall prediction process.

### 1. Data Analysis Report
  * This report focuses on the statistical analysis of the downloaded stock data. It explores key attributes of the 
    dataset, such as:
    * Price behavior (e.g., opening, closing, highs, and lows)
    * Volume traded
    * Time-series trends
  * The report also includes visualizations to provide a better understanding of how these features interact and their role in predicting stock price movements.
  * The goal of this analysis is to comprehend the stock market data fully before applying any models, ensuring we understand how each feature contributes to the overall price prediction.
  * [View Data Analysis Report](https://github.com/AHMEDNUSARI/Dataguess_TeknolojiTask/blob/main/Reports/Data_Analysis.ipynb)
### 2. Feature Engineering & Indicator Application Report
  * This report delves into the feature engineering process, where we apply technical indicators to the raw stock data to enhance prediction accuracy. It involves:
    * Applying various indicators such as MA_10, MACD, RSI, BB_High, BB_Low, etc.
    * Explaining each indicator’s role in stock price prediction, like how Moving Averages smooth out fluctuations, RSI signals overbought or oversold conditions, and MACD shows momentum trends.
  * The report also includes plots of each indicator, demonstrating their impact on price prediction and their crucial role in understanding market trends.
  * [View Feature Engineering Report](https://github.com/AHMEDNUSARI/Dataguess_TeknolojiTask/blob/main/Reports/Feature_engineer_report.ipynb)
### 3. Model Structure Report
  * This report provides a detailed explanation of the two models used in the project:
    * **Linear Regression:** A classic approach that assumes a linear relationship between input features and target stock prices.
    * **LSTM with Attention Mechanism:** A deep learning model that leverages long-term dependencies in time-series data and applies an attention mechanism to focus on the most relevant past data points.
  * The report details the architecture and working principles of each model, as well as the reasons for selecting these models in the context of stock price prediction.
  * [View Model Structure Report](https://github.com/AHMEDNUSARI/Dataguess_TeknolojiTask/blob/main/Reports/Models_Report.ipynb)
    
## 📊 Evaluation Metrics
For evaluating model performance, we prefer to use the R² (Coefficient of Determination) metric. Here's why R² is often more useful than RMSE and MAPE:

  * **R²** measures how well the model's predictions match the actual data. It is a better indicator of the model's explanatory power. A higher R² indicates that a larger proportion of the variance in stock prices is explained by the model, making it easier to compare different models in terms of how well they fit the data.
RMSE (Root Mean Square Error) and MAPE (Mean Absolute Percentage Error) are often used to assess the accuracy of predictions, but they can be misleading in certain scenarios:
  * **RMSE** gives more weight to large errors, which might distort the model's performance when there are significant outliers in the data.
  * **MAPE** can be biased when actual values are close to zero, leading to misleading results for certain types of data, such as stock prices with very low values at certain times.
Thus, **R²**  is preferred for stock price prediction because it provides a more comprehensive view of the model’s ability to explain the variability in the stock prices, without being heavily affected by outliers or extreme values.



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


### 2. Reusable Model Training for Custom Datasets
  * Script Name: TrainingCustomData.py
  * #### Steps to Run:
    1. Run the script using the following command:
        python TrainingCustomData.py
    2. The user will be prompted to:
      * Select the company from Yahoo Finance data (e.g., AAPL, TSLA, MSFT, AMZN, GOOGL) and specify the start and end date (e.g., "2000-05-05" to "2025-02-01").
      * Choose the features and indicators you want to apply. Available options include:
        ['Open', 'High', 'Low', 'Volume', 'MA_10', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR', 'OBV'].
      * Select the time-step sequence. The default is 10, but you can modify this to any value as needed.
    3. Finally, the user will be asked to select the model type:
      * 0: Linear Regression
      * 1: LSTM with Attention Mechanism
    4. After selecting the model, the script will train the model on the provided dataset, visualize the results, and display the evaluation metrics:
      * RMSE, MAPE, and R².
