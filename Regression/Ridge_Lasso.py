import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

n = 100

area = np.random.randint(500, 2000, n)
bedrooms = np.random.randint(1, 5, n)
age = np.random.randint(1, 30, n)
distance = np.random.randint(1, 20, n)

dummy = np.random.rand(n)
price = (
    50 * area +
    10000 * bedrooms -
    2000 * age -
    3000 * distance +
    np.random.randn(n) * 10000
)

df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "age": age,
    "distance": distance,
    "dummy": dummy
})

X = df
y = price

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)
ridge_pred = ridge.predict(x_test)

print("\nMSE:", mean_squared_error(y_test, ridge_pred))
print("R2 Score:", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.1)
lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)

print("\nMSE:", mean_squared_error(y_test, lasso_pred))
print("R2 Score:", r2_score(y_test, lasso_pred))