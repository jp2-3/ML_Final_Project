import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('database_24_25.csv')
tatum_df = df[df['Player'] == 'Jayson Tatum']
features = ['MP', 'FGA', '3PA', '3P%', 'FTA', 'TRB', 'AST', 'STL', 'BLK', 'TOV']

X = tatum_df[features]
y = tatum_df['PTS']

# Minutes Played vs Points Scored
plt.scatter(tatum_df['MP'], tatum_df['PTS'], color='purple')
plt.title('Minutes Played vs Points Scored')
plt.xlabel('Minutes Played')
plt.ylabel('Points Scored')
plt.show()

# Free throw average vs Points Scored
plt.scatter(tatum_df['FTA'], tatum_df['PTS'], color='red')
plt.title('FTA vs Points Scored')
plt.xlabel('Free Throw Average')
plt.ylabel('Points Scored')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# predictions
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Testing linear regression
print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, lr_preds))
print("R²:", r2_score(y_test, lr_preds))

# Testing random Forest
print("Random Forest Regressor:")
print("MSE:", mean_squared_error(y_test, rf_preds))
print("R²:", r2_score(y_test, rf_preds))

# plot linear regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_preds, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Linear Regression: Actual vs Predicted Points')
plt.show()

# plot random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_preds, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Random Forest: Actual vs Predicted Points')
plt.show()

