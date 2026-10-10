# Deep-Learning Autoencoders for Industrial Anomaly Detection

This repository contains the implementation and experiment record for a Master's thesis on anomaly detection in multivariate industrial vibration data. The study develops and evaluates recurrent autoencoders that learn normal operating behaviour and identify deviations through reconstruction error.

## Research objective

The objective is to build a data-driven detection workflow for three-axis sensor signals and to compare three recurrent architectures:

- Long Short-Term Memory (LSTM) autoencoder
- Gated Recurrent Unit (GRU) autoencoder
- Simple Recurrent Neural Network (RNN) autoencoder

The workflow is designed for industrial monitoring, predictive maintenance, and fault-detection research.

## Methodology

The implementation follows this sequence:

1. Load normal and anomalous three-axis vibration measurements.
2. Apply rolling-window smoothing to reduce short-term measurement variation.
3. Normalize each signal using min–max scaling.
4. Construct fixed-length temporal sequences.
5. Train each autoencoder using normal operating data.
6. Compute reconstruction errors for normal and anomalous sequences.
7. Select a detection threshold from the reconstruction-error distribution.
8. Classify observations and report accuracy, precision, recall, F1 score, and precision–recall curves.

This design separates representation learning from anomaly decision-making and provides the same evaluation procedure for all three model families.

## Repository artifacts

- [Readable Jupyter notebook](./Final%20article%20anomalies%20detection.ipynb) — valid notebook containing the implementation and recorded experiment cells.
- [Original PDF code export](./Final%20anomalies%20detection%20using%20deep%20learning%20models%20code.pdf) — preserved source export.
- [Readable PDF document](./Master%20thesis%20anomaly%20detection%20readable.pdf) — paginated version of the code and recorded outputs.

The notebook uses the dataset files defined in its input-path cells. For a new environment, replace those paths with the local locations of the corresponding normal and anomalous measurements.

## Evaluation

The experiment reports:

- Accuracy
- Precision
- Recall
- F1 score
- Reconstruction-error distributions
- Precision–recall curves
- Model-level performance visualizations

The same preprocessing, sequence construction, thresholding, and metric definitions are applied across the LSTM, GRU, and SimpleRNN experiments.

## Reproducibility

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

Use Python 3.8 or newer, update the dataset paths in the notebook, and execute the cells in order. The readable PDF provides a static record of the implementation and reported outputs.

## Academic context

This repository supports the Master's thesis project of Babar Hussain in machine learning, deep learning, and industrial anomaly detection. It is provided for academic research and reproducibility.

## License

The materials are intended for academic and research use.
