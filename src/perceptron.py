"""
This script implements and compares the custom Perceptron with Scikit-Learn's Perceptron.
It trains both models on multiple datasets after normalization and splitting.
The script outputs accuracy, final weights, and threshold for each dataset.

The perceptron performance on four datasets shows different degrees of linear separability:
- For Data_1.csv with accuracies for custom Perceptron 72% and SKLearn Perceptron 68%, this relatively low accuracy suggests that Data_1 is not linearly separable, as a linear classifier struggles to separate the classes cleanly.
- For Data_2.csv with accuracies for custom Perceptron 84%  and SKLearn Perceptron 80%, this moderate accuracy implies Data_2 is also not fully linearly separable, there may be some overlap between classes, since a perfect linear separation is not achieved.
- For Data_3.csv with accuracies for custom Perceptron 95% and SKLearn Perceptron 95%, the high accuracy indicates Data_3 is almost linearly separable, with only a few misclassified points.
- For Data_4.csv with accuracies for custom Perceptron 100%  and SKLearn Perceptron 100%, the perfect accuracy means Data_4 is linearly separable and the linear decision boundary can completely separate the classes with no errors.
In conclusion, the high accuracy for dataset- data_4.csv strongly suggests that the dataset is linearly separable. The Data_4.csv can be divided by a straight line with minimal or no misclassifications.

Date: March 25th, 2025
Author: Jaskirat Kaur, 000904397
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron as SKPerceptron

# Custom Perceptron Class
class Perceptron:
    """
    The implementation of the Perceptron learning algorithm.

    Attributes:
        learning_rate (float): The step size for weight updates.
        epochs (int): The number of iterations over the training data.
        weights (numpy.ndarray): The weight vector for the model.
        threshold (float): The bias term.
    """
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.threshold = 0  # Bias term

    def fit(self, features, labels):
        """
        Train the Perceptron using the given features and labels.

        Args:
            features (numpy.ndarray): The input data.
            labels (numpy.ndarray): The target labels (0 or 1).
        """
        num_features = features.shape[1]
        self.weights = np.zeros(num_features)
        self.threshold = 0

        for _ in range(self.epochs):
            for i in range(len(features)):
                activation = np.dot(features[i], self.weights) - self.threshold
                prediction = 1 if activation > 0 else 0
                error = labels[i] - prediction

                # Update rule
                self.weights += self.learning_rate * error * features[i]
                self.threshold -= self.learning_rate * error

    def predict(self, features):
        """
        Predict labels for given input features.
        Args:
            features (numpy.ndarray): The input data.
        Returns:
            numpy.ndarray: Predicted labels (0 or 1).
        """
        activations = np.dot(features, self.weights) - self.threshold
        return np.where(activations > 0, 1, 0)

    def accuracy(self, features, labels):
        """
        To calculate the accuracy of the model.
        Args:
            features (numpy.ndarray): The input data.
            labels (numpy.ndarray): The actual labels.
        Returns:
            float: Accuracy percentage.
        """
        predictions = self.predict(features)
        return np.mean(predictions == labels) * 100  # Convert to percentage

# Define dataset file paths
file_paths = {
    "Data_1.csv": "../Dataset/Data_1.csv",
    "Data_2.csv": "../Dataset/Data_2.csv",
    "Data_3.csv": "../Dataset/Data_3.csv",
    "Data_4.csv": "../Dataset/Data_4.csv"
}

# Dictionary to store results
results = {}

for file_name, path in file_paths.items():
    # Load dataset (skip header row if present)
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    # Separate features and labels
    features = data[:, :-1]
    labels = data[:, -1]

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split into training (80%) and test (20%)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train Custom Perceptron
    custom_perceptron = Perceptron(learning_rate=0.1, epochs=100)
    custom_perceptron.fit(features_train, labels_train)
    accuracy_custom = custom_perceptron.accuracy(features_test, labels_test)

    # Train SKLearn Perceptron
    sklearn_perceptron = SKPerceptron(max_iter=1000, tol=1e-3, random_state=42)
    sklearn_perceptron.fit(features_train, labels_train)
    accuracy_sklearn = sklearn_perceptron.score(features_test, labels_test) * 100  # Convert to percentage

    # Store results
    results[file_name] = {
        "custom_accuracy": accuracy_custom,
        "custom_weights": custom_perceptron.weights,
        "custom_threshold": custom_perceptron.threshold,
        "sklearn_accuracy": accuracy_sklearn,
        "sklearn_weights": sklearn_perceptron.coef_[0],
        "sklearn_threshold": -sklearn_perceptron.intercept_[0]
    }

# Print results
print("\n===== Perceptron Results =====\n")
for file, data in results.items():
    print(f"{file}:\n")
    print(f"  Custom Perceptron -> Accuracy: {data['custom_accuracy']:.0f}% | W: {data['custom_weights'].tolist()} | T: {data['custom_threshold']:.1f}\n")
    print(f"  SKLearn Perceptron -> Accuracy: {data['sklearn_accuracy']:.0f}% | W: {data['sklearn_weights'].tolist()} | T: {data['sklearn_threshold']:.1f}")
    print("-" * 80)
