
# Anomaly Detection Using Deep Learning Autoencoders (LSTM, GRU, RNN)

This repository contains a **Jupyter Notebook implementation of anomaly detection** using **deep learning autoencoders** applied to multivariate time-series data.  
The project compares **LSTM, GRU, and Simple RNN autoencoders** for detecting anomalies using reconstruction error.

---

## 📌 Project Overview

The goal of this project is to detect anomalous behavior in multivariate sensor data by learning normal patterns through sequence-based autoencoders.

Key highlights:
- Sliding-window time-series modeling
- LSTM, GRU, and RNN autoencoders
- Threshold-based anomaly detection
- Extensive visualizations and performance evaluation


## 🧠 Models Implemented

- LSTM Autoencoder  
- GRU Autoencoder  
- Simple RNN Autoencoder  

All models are trained **only on normal data** and anomalies are detected using reconstruction error statistics.

## 📂 Repository Structure


├── code_converted.ipynb     # Main Jupyter Notebook

---

## 🔧 Requirements

Python 3.8+

Main dependencies:
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow / keras

Install all dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```
---

## 📊 Methodology

1. Data preprocessing (smoothing + normalization)
2. Sequence generation using sliding windows
3. Autoencoder training on normal data
4. Reconstruction error computation
5. Threshold-based anomaly classification
6. Performance evaluation

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Precision–Recall curves

All metrics and plots are generated inside the notebook.

---

## 🚀 Applications

- Predictive maintenance
- Industrial IoT monitoring
- Fault detection systems
- Time-series anomaly detection

---


## 👤 Author

**Babar Hussain**  
PhD Researcher – Computer Science  
Research focus: Machine Learning, Edge-AI, Anomaly Detection



## 📜 License

This project is intended for **academic and research use**.
