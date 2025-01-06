import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import argparse
from datetime import datetime

#using range=10 for the range of the wine quality metric
range = 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for the dataset')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for the dataset split')
    parser.add_argument("--save", choices=["true", "false"], default="false")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    test_size = args.test_size
    random_state = args.random_state
    save = args.save
    if save == 'true':
        save = 1
    if save == 'false':
        save = 0

# Load the wine quality dataset
data = pd.read_csv('winequality-red.csv')
std_dev = data['quality'].std()

# Split the data into features (X) and target variable (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Create a Decision Tree Regressor
tree = DecisionTreeRegressor(random_state=42)

# Train the Decision Tree Regressor
tree.fit(X_train, y_train)

# Predict the wine quality using the trained Decision Tree Regressor
y_pred = tree.predict(X_test)
if save:
    df = pd.DataFrame({
        'fixed acidity': X_test['fixed acidity'],
        'volatile acidity': X_test['volatile acidity'],
        'citric acid': X_test['citric acid'],
        'residual sugar': X_test['residual sugar'],
        'chlorides': X_test['chlorides'],
        'free sulfur dioxide': X_test['free sulfur dioxide'],
        'total sulfur dioxide': X_test['total sulfur dioxide'],
        'density': X_test['density'],
        'pH': X_test['pH'],
        'sulphates': X_test['sulphates'],
        'alcohol': X_test['alcohol'],
        'predicted quality': y_pred
    })
    filename = f"wineQuality-predictions_{test_size}_{random_state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

# Evaluate the performance of the Decision Tree Regressor
mse = mean_squared_error(y_test, y_pred)
percentage = (mse / std_dev) * 100
# Calculate the percentage of the range that the MSE represents
percentage_range = (mse / 10) * 100
print(f'MSE: {mse}')
print(f'standard deviation: {std_dev}')
print(f"The MSE of {mse} represents {percentage:.2f}% of the standard deviation.")
print(f"The MSE of {mse} represents {percentage_range:.2f}% of the range.")