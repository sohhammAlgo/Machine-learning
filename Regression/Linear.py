import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset (sufficient size)
X = np.array([
    [1, 6],
    [2, 7],
    [3, 8],
    [4, 9],
    [5, 10],
    [6, 11],
    [7, 12],
    [8, 13],
    [9, 14],
    [10, 15]
])

y = np.array([1,2,3,4,5,6,7,8,9,10])

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Debug check
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# 3. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Prediction
y_pred = model.predict(X_test)

# 5. Results
print("\nActual values:", y_test)
print("Predicted values:", y_pred)

print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 6. Optional: Visualization
import matplotlib.pyplot as plt

X_plot = X[:, 0]

# Sort for smooth line
sorted_idx = np.argsort(X_plot)
X_sorted = X_plot[sorted_idx]

# Predict on sorted data
y_line = model.predict(X[sorted_idx])

plt.scatter(X_plot, y, color='blue', label="Actual Data")
plt.plot(X_sorted, y_line, color='red', label="Regression Line")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.title("Linear Regression Visualization")
plt.legend()
plt.show()