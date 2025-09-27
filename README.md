# Microsoft Stock Price Forecasting with ANN, RNN, and LSTM

## Description

This project analyzes Microsoft stock prices using time series forecasting techniques with three neural network models: Artificial Neural Network (ANN), Recurrent Neural Network (RNN), and Long Short-Term Memory (LSTM). The notebook includes data preprocessing, anomaly detection, visualization, seasonal decomposition, model training, evaluation, and comparison of predictions.

## Requirements

- Python 3.10
- Libraries:
  - numpy
  - matplotlib
  - tensorflow
  - pandas
  - statsmodels
  - scikit-learn

Install the required libraries using pip:

```
pip install numpy matplotlib tensorflow pandas statsmodels scikit-learn
```

## Data

- `Microsoft_Stock.csv`: CSV file containing historical Microsoft stock data, including columns like 'Close' for closing prices.

## Usage

1. Ensure all dependencies are installed.
2. Open and run the Jupyter notebook `projet1.ipynb` in a Jupyter environment (e.g., Jupyter Notebook or JupyterLab).

The notebook will:
- Load and preprocess the data.
- Detect and visualize anomalies using IQR method.
- Perform seasonal decomposition.
- Scale features and prepare sequences for time series prediction (60-day lookback).
- Train ANN, RNN, and LSTM models.
- Evaluate and compare model performances.
- Plot predictions and true values.

## Models

- **ANN (Artificial Neural Network)**: A basic feedforward neural network.
- **RNN (Recurrent Neural Network)**: Handles sequential data but may struggle with long-term dependencies.
- **LSTM (Long Short-Term Memory)**: An advanced RNN variant designed for capturing long-term patterns in sequences.

Hyperparameters:
- Epochs: 50
- Batch size: 32
- Optimizer: Adam
- Loss: Mean Squared Error

## Results

The models are evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

Based on the analysis:
- **LSTM** performs best for time series forecasting due to its ability to handle long-term dependencies.
- **RNN** outperforms ANN in sequence handling but is less effective than LSTM.
- **ANN** is least suitable as it does not account for temporal relationships.

Visualizations include training/validation loss curves, anomaly detection plots, seasonal decomposition, and prediction comparisons.

## Conclusion

For accurate stock price forecasting, LSTM is the recommended model. The project demonstrates the importance of choosing appropriate architectures for time series data.


