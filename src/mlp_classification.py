import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

"""
This script evaluates the performance of Multi-Layer Perceptrons (MLPs) with Decision Trees
on multiple classification datasets, including the malware classification dataset (TUNADROMD).
It includes GridSearchCV tuning for both MLP and Decision Tree models for enhanced performance.

Special Cases:
- Data_1.csv uses GridSearchCV for MLP hyperparameter tuning.
- Data_4.csv uses GridSearchCV to find the best Decision Tree hyperparameters.

For Data_4.csv, I ran the GridSearchCV to optimize Decision Tree parameters, but even the best configuration
(max_depth=5, min_samples_split=10) only achieved ~70.8% accuracy. This further confirms that the dataset
is better handled by models capable of learning nonlinear boundaries, such as the MLP, which achieved 92% accuracy.

UCI Machine Repository Dataset Reference: https://archive.ics.uci.edu/dataset/813/tunadromd

Date: March 25th, 2025
Author: Jaskirat Kaur
"""

# Define dataset paths
file_paths = {
    "Data_1.csv": "../Dataset/Data_1.csv",
    "Data_2.csv": "../Dataset/Data_2.csv",
    "Data_3.csv": "../Dataset/Data_3.csv",
    "Data_4.csv": "../Dataset/Data_4.csv",
    "TUANDROMD.csv": "../Dataset/TUANDROMD.csv"
}

# Custom tuned configurations per file
tuned_configs = {
    "Data_1.csv": {"hidden_layers": (50, 25), "lr": 0.005},
    "Data_2.csv": {"hidden_layers": (40, 20), "lr": 0.005},
    "Data_3.csv": {"hidden_layers": (10,), "lr": 0.01},
    "Data_4.csv": {"hidden_layers": (5,), "lr": 0.01},
    "TUANDROMD.csv": {"hidden_layers": (100, 50), "lr": 0.001}
}

# Dictionary to store results
mlp_results = {}

for file_name, path in file_paths.items():
     # Load CSV file and separate features and target labels
    raw_data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=None, encoding='utf-8')
    features = np.array([row[:-1] for row in raw_data], dtype=float)
    target_raw = [row[-1] for row in raw_data]

    # Encode categorical labels if needed
    if file_name == "TUANDROMD.csv":
        labels = np.array([1 if str(label).lower() == 'malware' else 0 for label in target_raw], dtype=int)
    else:
        labels = np.array(target_raw, dtype=float)
        if not np.array_equal(labels, labels.astype(int)):
            labels = np.round(labels).astype(int)

    # Normalize feature
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Split dataset into training and test sets (80% train, 20% test)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Use tuned hyperparameters
    config = tuned_configs[file_name]
    layers = config["hidden_layers"]
    lr = config["lr"]

    # GridSearchCV for MLP (Data_1.csv)
    if file_name == "Data_1.csv":
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(features_train, labels_train)
        dt_preds = dt_model.predict(features_test)
        dt_acc = accuracy_score(labels_test, dt_preds) * 100

        # MLP tuning
        param_grid = {
            'hidden_layer_sizes': [(10,), (20,), (50,), (50, 25), (100,)],
            'learning_rate_init': [0.001, 0.005, 0.01],
            'activation': ['relu', 'tanh'],
            'solver': ['adam']
        }
        # Run GridSearchCV for MLP
        mlp_grid = MLPClassifier(max_iter=200, tol=1e-4, early_stopping=True,n_iter_no_change=10, random_state=42)

        # Run grid search for Decision Tree
        grid_search = GridSearchCV(mlp_grid, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(features_train, labels_train)

        # Best decision tree model
        best_model = grid_search.best_estimator_
        labels_pred = best_model.predict(features_test)
        test_acc = accuracy_score(labels_test, labels_pred) * 100

        # Store both MLP and DT results
        mlp_results[file_name] = {
            "type": "classification",
            "decision_tree_accuracy": dt_acc,
            "mlp_accuracy": test_acc,
            "mlp_hidden_layers": best_model.hidden_layer_sizes,
            "mlp_iterations": best_model.n_iter_,
            "mlp_learning_rate": best_model.learning_rate_init,
            "mlp_tolerance": best_model.tol
        }

    else:
        # Perform GridSearchCV for Decision Tree ONLY on Data_4.csv
        if file_name == "Data_4.csv":
            print("\n===== GridSearchCV for Decision Tree - Data_4.csv =====")

            # Define hyperparameter grid for decision tree
            param_grid_dt = {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }

            # Run grid search for Decision Tree
            grid_dt = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid_dt, cv=5, scoring='accuracy'
            )
            grid_dt.fit(features_train, labels_train)

            # Best decision tree model
            dt_best = grid_dt.best_estimator_
            dt_preds = dt_best.predict(features_test)
            dt_acc = accuracy_score(labels_test, dt_preds) * 100

            # Print best parameters and result
            print("Best DecisionTree Parameters:", grid_dt.best_params_)
            print(f"DecisionTree Accuracy (Tuned): {dt_acc:.2f}%")
        else:
            # Default decision tree training for other datasets
            dt_best = DecisionTreeClassifier(random_state=42)
            dt_best.fit(features_train, labels_train)
            dt_preds = dt_best.predict(features_test)
            dt_acc = accuracy_score(labels_test, dt_preds) * 100

        # Train MLPClassifier with tuned settings
        mlp = MLPClassifier(
            hidden_layer_sizes=layers,
            learning_rate_init=lr,
            max_iter=200,
            tol=1e-4,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42
        )

        mlp.fit(features_train, labels_train)
        mlp_preds = mlp.predict(features_test)
        mlp_acc = accuracy_score(labels_test, mlp_preds) * 100

        # Store results for MLP and DT
        mlp_results[file_name] = {
            "type": "classification",
            "decision_tree_accuracy": dt_acc,
            "mlp_accuracy": mlp_acc,
            "mlp_hidden_layers": mlp.hidden_layer_sizes,
            "mlp_iterations": mlp.n_iter_,
            "mlp_learning_rate": mlp.learning_rate_init,
            "mlp_tolerance": mlp.tol
        }


# Print results
print("\n===== Multi-Layer Perceptron Results (Tuned) =====\n")
for file, data in mlp_results.items():
    print(f"File: {file}")
    if data["decision_tree_accuracy"] is not None:
        print(f"  Decision Tree: {data['decision_tree_accuracy']:.0f}% Accuracy")
    print(f"  MLP: hidden layers = {data['mlp_hidden_layers']}, LR = {data['mlp_learning_rate']:.5f}, tol = {data['mlp_tolerance']:.7f}")
    print(f"  {data['mlp_accuracy']:.0f}% Accuracy, {data['mlp_iterations']} iterations")
    print("-" * 80)
