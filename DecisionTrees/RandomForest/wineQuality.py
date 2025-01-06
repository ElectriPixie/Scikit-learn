import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import argparse
from datetime import datetime

#using range=10 for the range of the wine quality metric
range = 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for the dataset')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for the dataset split')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators for the Random Forest')
    parser.add_argument("--save", choices=["true", "false"], default="false")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    test_size = args.test_size
    random_state = args.random_state
    n_estimators = args.n_estimators
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

# Create a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

# Train the Random Forest Regressor
rf.fit(x_train, y_train)

# Predict the wine quality using the trained Random Forest Regressor
y_pred = rf.predict(x_test)
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
        'predicted quality': x_pred
    })
    filename = f"wineQuality-predictions_{test_size}_{random_state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

# Evaluate the performance of the Random Forest Regressor
mse = mean_squared_error(y_test, y_pred)
percentage = (mse / std_dev) * 100
# Calculate the percentage of the range that the MSE represents
percentage_range = (mse / 10) * 100
print(f'MSE: {mse}')
print(f'standard deviation: {std_dev}')
print(f"The MSE of {mse} represents {percentage:.2f}% of the standard deviation.")
print(f"The MSE of {mse} represents {percentage_range:.2f}% of the range.")