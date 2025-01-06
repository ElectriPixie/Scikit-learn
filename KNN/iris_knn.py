import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int,
                        default=5, help="Number of neighbors")
    parser.add_argument(
        "--average", choices=["macro", "weighted", "micro"], default="macro", help="Type of averaging")
    parser.add_argument("--save", choices=["true", "false"], default="false")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(f"n_neighbors: {args.n_neighbors}, average: {
          args.average}, save: {args.save}")
    n_neighbors = args.n_neighbors
    average = args.average
    save = args.save
    if save == 'true':
        save = 1
    if save == 'false':
        save = 0

iris = load_iris()
X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
if save:
    # Save the results
    print("Saving results...")
    df = pd.DataFrame({
        'Actual Label': y_test,
        'Predicted Label': y_pred
    })
    filename = f"iris_knn_{n_neighbors}_{average}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=average)
recall = recall_score(y_test, y_pred, average=average)
f1 = f1_score(y_test, y_pred, average=average)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
