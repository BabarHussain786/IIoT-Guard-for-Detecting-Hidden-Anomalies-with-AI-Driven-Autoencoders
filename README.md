# Anomaly Detection Using Deep Learning Autoencoders

Master's thesis project for multivariate industrial vibration anomaly detection using sequence autoencoders.

## Project scope

The study evaluates LSTM, GRU, and SimpleRNN autoencoders on three-axis vibration signals. The workflow includes rolling-window smoothing, min–max normalization, sequence construction, reconstruction-error thresholding, and classification metrics.

## Files

- [Readable Jupyter notebook](./Final%20article%20anomalies%20detection.ipynb) — valid notebook JSON reconstructed from the original export.
- [Original PDF code export](./Final%20anomalies%20detection%20using%20deep%20learning%20models%20code.pdf) — preserved source document.
- [Readable PDF document](./Master%20thesis%20anomaly%20detection%20readable.pdf) — paginated version with normal code and output spacing.

The readable notebook and PDF preserve the recorded code and outputs. They do not rerun the experiments automatically; the original dataset paths in the notebook must be adjusted for a new machine.

## Methodology

1. Load normal and anomalous three-axis sensor data.
2. Smooth each signal with a rolling window.
3. Normalize the signals and create time-series windows.
4. Train LSTM, GRU, and SimpleRNN autoencoders on normal behavior.
5. Detect anomalies using reconstruction error and a percentile threshold.
6. Report accuracy, precision, recall, F1 score, and precision–recall curves.

## Reported metrics

The recorded experiment reports accuracy, precision, recall, and F1 score for each model. The full values and generated outputs are retained in the notebook and readable PDF.

## Requirements

Python 3.8 or newer with:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Academic context

This repository contains material from Babar Hussain's Master's thesis project and is intended for academic and research use.
