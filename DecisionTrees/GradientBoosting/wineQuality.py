import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import argparse
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for the dataset')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for the dataset split')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators for the models')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the models')
    parser.add_argument("--save", choices=["true", "false"], default="false")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    test_size = args.test_size
    random_state = args.random_state
    n_estimators = args.n_estimators
    learning_rate = args.learning_rate
    save = args.save
    if save == 'true':
        save = 1
    if save == 'false':
        save = 0

#using range=10 for the range of the wine quality metric
range = 10

# Load the wine quality dataset
data = pd.read_csv('winequality-red.csv')
std_dev = data['quality'].std()

# Split the data into features (X) and target variable (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Create a Gradient Boosting Regressor
gb = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

# Train the models
gb.fit(X_train, y_train)

# Predict the wine quality using the trained models
gb_pred = gb.predict(X_test)

# Evaluate the performance of the models
gb_mse = mean_squared_error(y_test, gb_pred)

percentage_gb = (gb_mse / std_dev) * 100
percentage_gb_range = (gb_mse / 10) * 100

print(f"Gradient Boosting MSE: {gb_mse}")
print(f"The MSE of Gradient Boosting {gb_mse} represents {percentage_gb:.2f}% of the standard deviation.")
print(f"The MSE of Gradient Boosting {gb_mse} represents {percentage_gb_range:.2f}% of the range.")

if save:
    df = pd.DataFrame({
        'fixed acidity': X_test['fixed acidity'],
        'volatile acidity': X_test['volatile acidity'],
        'citric acid': X_test['citric acid'],
        'residual sugar': X_test['residual sugar'],
        'chlorides': X_test['chlorides'],
        'free sulfur Dioxide': X_test['free sulfur Dioxide'],
        'total Sulfur Dioxide': X_test['total Sulfur Dioxide'],
        'density': X_test['density'],
        'pH': X_test['pH'],
        'sulfates': X_test['sulfates'],
        'alcohol': X_test['alcohol'],
        'predicted quality': gb_pred
    })
    filename = f"wineQuality-predictions_{test_size}_{random_state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)