# Stock Market Analysis using RNN & LSTM

CS 4375.001 - Machine Learning Final Project  
University of Texas at Dallas

## Team Members

- Thuyan Dang (THD210004) - Training Pipeline & Evaluation
- Nayah Sayo (NXS210108) - RNN Implementation & Data Preprocessing  
- Nidhi Majoju (NXM220069) - Project Coordination & Documentation
- Aryan Neeli (AKN220008) - LSTM Implementation

## Project Overview

This project implements Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models from scratch to predict stock prices using historical market data. Both architectures are built using only NumPy, with no reliance on high-level deep learning frameworks for the core algorithms. The goal is to compare how well each model captures temporal patterns in financial time series and evaluate their effectiveness for next-day stock price prediction.

The models process 30-day sequences of stock market features and predict the following day's closing price. We trained 24 different configurations (12 RNN + 12 LSTM) across various hyperparameters to identify optimal settings.

## Technical Implementation

### RNN Architecture
The vanilla RNN processes sequences timestep-by-timestep using a single hidden layer with tanh activation. While effective at capturing short-term patterns, RNNs struggle with long-term dependencies due to the vanishing gradient problem during backpropagation through time.

### LSTM Architecture  
The LSTM addresses RNN limitations through a gating mechanism with four specialized components: forget gate, input gate, cell state candidate, and output gate. The cell state acts as a memory highway that allows gradients to flow through time without significant degradation, enabling the network to learn both short and long-term dependencies.

Both models are trained using:
- Backpropagation Through Time (BPTT)
- Stochastic Gradient Descent optimization
- Gradient clipping (max norm = 1.0) to prevent exploding gradients
- Mean Squared Error (MSE) loss function

## Dataset

We use 5 years of historical stock data for Apple Inc. (AAPL) from January 1, 2018 to December 31, 2022, downloaded via the Yahoo Finance API. The dataset contains approximately 1,250 daily trading observations.

Features:
- Open price
- High price  
- Low price
- Close price
- Trading volume

All features are normalized to the range [0, 1] using Min-Max scaling. Sequences of 30 consecutive trading days are used as input, with the goal of predicting the closing price on day 31. The data is split 80/20 for training and testing.

## Repository Structure

```
CS4375-Project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── raw_data.parquet
│   └── preprocessed/
│       ├── preprocessed_data.parquet
│       └── preprocessed_data.csv
│
├── src/
│   ├── data_loader.py      # downloads stock data from Yahoo Finance
│   ├── preprocess.py        # data normalization and sequence generation
│   ├── rnn.py              # RNN implementation from scratch
│   ├── lstm.py             # LSTM implementation from scratch
│   ├── train.py            # training pipeline and experiments
│   └── utils.py            # helper functions
│
├── models/                  # saved model weights (.npz format)
│   ├── RNN_hs16_lr0.005_ep30.npz
│   ├── LSTM_hs16_lr0.005_ep50.npz
│   └── ...
│
└── results/                 # training outputs and visualizations
    ├── training_output.log
    ├── RNN_mse_comparison.png
    ├── LSTM_mse_comparison.png
    ├── training loss curves
    └── prediction plots
```

## Installation

Requires Python 3.8 or higher.

Clone the repository:
```bash
git clone https://github.com/NaSa30/CS4375-Project.git
cd CS4375-Project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages include numpy, pandas, matplotlib, scikit-learn, torch (for DataLoader only), and yfinance.

## Usage

### Running the full training pipeline

To train all 24 model configurations:

```bash
python src/train.py
```

This will train 12 RNN variants and 12 LSTM variants across different hyperparameter combinations, save all models to the models directory, generate visualizations in results, and log complete training details to results/training_output.log.

Training typically takes 30-60 minutes depending on hardware.

### Downloading data

To download AAPL stock data separately:

```bash
python src/data_loader.py
```

### Preprocessing data

To generate normalized sequences:

```bash
python src/preprocess.py
```

## Hyperparameter Grid

Both RNN and LSTM models are evaluated across these configurations:

- Hidden sizes: 16, 32
- Learning rates: 0.001, 0.005  
- Training epochs: 20, 30, 50

This results in 12 configurations per model type (2 x 2 x 3 = 12), for a total of 24 trained models.

Fixed parameters:
- Input size: 5 features
- Output size: 1 (next day closing price)
- Sequence length: 30 days
- Batch size: 16

## Results

### Best Model Performance

After training and evaluating all configurations, the top performers were:

**RNN:**
- Configuration: RNN with 16 hidden units, learning rate 0.005, trained for 30 epochs
- Test MSE: 225.15
- Test MAE: 12.19

**LSTM:**  
- Configuration: LSTM with 16 hidden units, learning rate 0.005, trained for 50 epochs
- Test MSE: 203.76
- Test MAE: 11.74

The LSTM outperformed RNN by approximately 9% in MSE and 4% in MAE, demonstrating its superior ability to capture temporal dependencies in stock price data.

### Key Observations

Training loss curves show rapid initial decrease followed by plateau around 25 epochs for both architectures. The LSTM benefits from extended training (50 epochs) due to its more complex gating structure, while the RNN tends to overfit beyond 30 epochs.

Smaller hidden sizes (16 units) consistently outperformed larger sizes (32 units), suggesting that the additional capacity leads to overfitting on this dataset. Similarly, the moderate learning rate of 0.005 provided the best balance - 0.001 was too conservative while 0.01 caused unstable training.

Both models show limitations including one-step prediction lag and conservative predictions during volatile market periods. The training period includes the COVID-19 pandemic (2020-2021), a time of unprecedented market turbulence that likely contributes to higher prediction errors.

### Visualizations

The results directory contains comprehensive visualizations for each model:

- Training loss curves showing convergence behavior
- Prediction vs actual price comparisons on test data
- R-squared comparisons between training and test sets

## Implementation Details

### RNN Forward Pass

The RNN computes hidden states sequentially:

```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
y = W_hy * h_T + b_y
```

### LSTM Forward Pass

The LSTM uses a gating mechanism:

```
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)      forget gate
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)      input gate  
c_tilde = tanh(W_c * [h_{t-1}, x_t] + b_c)     candidate cell
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)      output gate
c_t = f_t * c_{t-1} + i_t * c_tilde            cell update
h_t = o_t * tanh(c_t)                          hidden state
```

Both implementations use backpropagation through time for gradient computation. The additive cell state update in LSTM allows gradients to flow through time with minimal degradation, addressing the vanishing gradient problem.

## Evaluation Metrics

**Mean Squared Error (MSE):** Computed as the average squared difference between predicted and actual prices. This metric emphasizes larger errors due to the squaring operation.

**Mean Absolute Error (MAE):** Computed as the average absolute difference between predictions and actual values, providing a more interpretable measure of average prediction deviation.

**R-squared:** Measures the proportion of variance in the target variable explained by the model, useful for detecting overfitting when comparing training and test performance.

## Academic Context

This project was completed as the final project for CS 4375 - Introduction to Machine Learning at the University of Texas at Dallas in Fall 2024. The project requirements included implementing a machine learning technique from scratch without using built-in libraries for the core algorithm, applying it to a real-world dataset, and producing a comprehensive analysis with visualizations.

## Contact

For questions about this project, contact the team members at their UTD email addresses.

Last updated: November 28, 2024
