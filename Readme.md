# Neural Network Classification

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

#### Perceptron Results

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Custom Perceptron Accuracy</th>
      <th>Custom T</th>
      <th>SKLearn Perceptron Accuracy</th>
      <th>SKLearn T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data_1.csv</td>
      <td>72%</td>
      <td>2.4</td>
      <td>68%</td>
      <td>24.0</td>
    </tr>
    <tr>
      <td>Data_2.csv</td>
      <td>84%</td>
      <td>2.9</td>
      <td>80%</td>
      <td>30.0</td>
    </tr>
    <tr>
      <td>Data_3.csv</td>
      <td>95%</td>
      <td>0.0</td>
      <td>95%</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <td>Data_4.csv</td>
      <td>100%</td>
      <td>-0.2</td>
      <td>100%</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>

#### MLP & Decision Tree Results

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Decision Tree Accuracy</th>
      <th>MLP Hidden Layers</th>
      <th>MLP Learning Rate</th>
      <th>MLP Tolerance</th>
      <th>MLP Accuracy</th>
      <th>MLP Iterations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data_1.csv</td>
      <td>96%</td>
      <td>(100,)</td>
      <td>0.01</td>
      <td>0.0001</td>
      <td>92%</td>
      <td>14</td>
    </tr>
    <tr>
      <td>Data_2.csv</td>
      <td>96%</td>
      <td>(40, 20)</td>
      <td>0.005</td>
      <td>0.0001</td>
      <td>100%</td>
      <td>19</td>
    </tr>
    <tr>
      <td>Data_3.csv</td>
      <td>82%</td>
      <td>(10,)</td>
      <td>0.01</td>
      <td>0.0001</td>
      <td>91%</td>
      <td>15</td>
    </tr>
    <tr>
      <td>Data_4.csv</td>
      <td>71%</td>
      <td>(5,)</td>
      <td>0.01</td>
      <td>0.0001</td>
      <td>92%</td>
      <td>21</td>
    </tr>
    <tr>
      <td>TUANDROMD.csv</td>
      <td>100%</td>
      <td>(100, 50)</td>
      <td>0.001</td>
      <td>0.0001</td>
      <td>100%</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
