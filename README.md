# 📈 Stock Market Analysis using RNN & LSTM
### CS 4375.001 — Final Project

## 👥 Team Members
- Thuyan Dang — THD210004  
- Nayah Sayo — NXS210108  
- Nidhi Majoju — NXM220069  
- Aryan Neeli — AKN220008  

---

## 📝 Project Overview
This project explores the use of Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models to predict stock prices using historical market data. LSTMs are particularly effective for time-series forecasting due to their ability to capture long-term patterns and dependencies in sequential data. Our goal is to compare the performance of a vanilla RNN and an LSTM built **from scratch** and evaluate their ability to predict the next day’s closing price for Apple (AAPL).

---

## 🧠 Techniques & Algorithms
- Programming Language: **Python**
- Built from scratch:
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
- Training with **Backpropagation Through Time (BPTT)**
- Optimized using **Gradient Descent**
- Loss function: **Mean Squared Error (MSE)**
- Prediction task: **Sequence-to-One**
  - Input: Previous **30 days** of stock prices  
  - Output: Next day's closing price  

---

## 📊 Dataset
- **Source:** Yahoo Finance  
- **Stock:** Apple (AAPL)  
- **Time Range:** Jan 1, 2018 – Dec 31, 2022  
- **~1250 daily observations**  
- **Features Used:** Open, High, Low, Close, Volume  
- Data normalized to [0, 1]  
- **Train/Test Split:** 80% / 20%

---

## 🔧 Repository Structure
stock-market-lstm/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   └── AAPL_2018_2022.csv
│   └── processed/
│       └── train_test_split.pkl
|   |__ parquetToCSV.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_rnn_from_scratch.ipynb
│   ├── 04_lstm_from_scratch.ipynb
│   ├── 05_training_and_results.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── rnn.py
│   ├── lstm.py
│   ├── train.py
│   └── utils.py
│
├── models/
│   ├── rnn_weights.pth
│   ├── lstm_weights.pth
│
├── results/
│   ├── rnn_predictions.png
│   ├── lstm_predictions.png
│   └── metrics.txt
│
├── requirements.txt
└── .gitignore

## ▶️ How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt

2. python src/train.py