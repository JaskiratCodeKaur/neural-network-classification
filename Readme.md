# Neural Network Classification & Image Analysis

This project implements classical neural network models for classification tasks, including:
- **Custom Perceptron** vs **Scikit-Learn Perceptron**
- **Multi-Layer Perceptron (MLP)** with hyperparameter tuning
- Comparison with **Decision Tree classifiers** on selected datasets

## Features
- Perceptron: Custom implementation with weights and threshold updates
- MLP: GridSearchCV tuning for hidden layers, learning rate, and activation
- Decision Tree comparison to highlight nonlinear separability
- Preprocessing with normalization and train-test split
- Evaluation with accuracy metrics

## Technologies
- Python
- NumPy
- Scikit-Learn

## Datasets
- `Dataset/Data_1.csv` to `Data_4.csv`
- `Dataset/TUANDROMD.csv` (malware classification)

## Usage
1. Clone the repository:
```bash
git clone https://github.com/JaskiratCodeKaur/neural-network-image-classification.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the scripts:
```bash
python src/perceptron.py
python src/mlp_processor.py
```
## Results
- Perceptron achieved up to 100% accuracy on linearly separable datasets
- MLP achieved up to 92% accuracy on nonlinear datasets
- Decision Tree accuracy is lower for datasets that require nonlinear boundaries