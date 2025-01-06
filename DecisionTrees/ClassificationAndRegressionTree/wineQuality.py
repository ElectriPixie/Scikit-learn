import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
x = data.drop('quality', axis=1)
y = data['quality']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

# Create a Decision Tree Regressor
tree = DecisionTreeRegressor(random_state=42)

# Train the Decision Tree Regressor
tree.fit(x_train, y_train)

# Predict the wine quality using the trained Decision Tree Regressor
y_pred = tree.predict(x_test)
if save:
    df = pd.DataFrame({
        'fixed acidity': x_test['fixed acidity'],
        'volatile acidity': x_test['volatile acidity'],
        'citric acid': x_test['citric acid'],
        'residual sugar': x_test['residual sugar'],
        'chlorides': x_test['chlorides'],
        'free sulfur dioxide': x_test['free sulfur dioxide'],
        'total sulfur dioxide': x_test['total sulfur dioxide'],
        'density': x_test['density'],
        'pH': x_test['pH'],
        'sulphates': x_test['sulphates'],
        'alcohol': x_test['alcohol'],
        'predicted quality': y_pred
    })
    filename = f"wineQuality-predictions_{test_size}_{random_state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

# Evaluate the performance of the Decision Tree Regressor
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")