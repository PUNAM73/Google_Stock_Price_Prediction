# Google_Stock_Price_Prediction

# Overview
This project focuses on predicting Google’s stock prices using Long Short-Term Memory (LSTM) neural networks. By leveraging historical stock price data, the model aims to forecast future trends, providing valuable insights for investors and analysts.

# Key Features
1. Data Preprocessing:

  Scaled stock price data using MinMaxScaler for efficient model training.
  
  Created a data structure with 60 timesteps to predict the next timestep.

2. LSTM Model Architecture:

  Four LSTM layers with 50 units each, followed by dropout layers to reduce overfitting.
  
  Fully connected dense layer for final prediction.
  
  Optimized with Adam optimizer and mean squared error (MSE) loss function.

3. Prediction and Visualization:

  Predicted stock prices for the test set and visualized the results alongside real prices.
  
  Achieved an RMSE of 10.98 and accuracy of 89.02%.
# Technologies Used
Programming Language: Python

Libraries: NumPy, Pandas, Matplotlib, Scikit-learn, Keras

Development Environment: Google Colab, Jupyter Notebook
# Project Workflow
1. Data Collection: Loaded historical stock price data for training and testing.
2. Data Preprocessing: Scaled data, created timesteps, and reshaped input for LSTM compatibility.
3. Model Training: Built and trained an LSTM model with dropout regularization.
4. Prediction: Forecasted Google’s stock prices and compared predictions to real values.
5. Evaluation: Measured model performance using RMSE and visualized results.
# Results
The model successfully captured stock price trends with 89.02% accuracy, demonstrating its effectiveness in time series forecasting.

# Future Improvements
1. Experiment with different window sizes and hyperparameters to improve accuracy.
2. Incorporate external financial indicators for enhanced predictions.
3. Deploy the model using Flask or Streamlit for real-time stock prediction.
