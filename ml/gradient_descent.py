#%%
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

#%%
df = pd.DataFrame(
    {
        "area": [2600, 3000, 3200, 3600, 4000],
        "price": [550_000, 565_000, 610_000, 680_000, 725_000],
    }
)
# df = pd.DataFrame(
#     {
#         "area": [1, 2, 3, 4, 5],
#         "price": [5, 7, 9, 11, 13],
#     }
# )
df
#%%
df.describe()
df.info()
df.shape
df.head(2)
#%%
sns.pairplot(df, x_vars="area", y_vars="price", size=5, aspect=1.15, kind="reg")

#%%
model = LinearRegression()
model.fit(df[["area"]], df["price"])
#%%
plt.scatter(df["area"], df["price"])
plt.plot(df["area"], model.predict(df[["area"]]), color="red")
#%%
# joblib.dump(model, "ml_model")
#%%
model.coef_, model.intercept_
#%%
x, y = df["area"], df["price"]
m, b = 0, 0
n, learning_rate, iterations = df.shape[0], 0.000000001, 10
for i in range(iterations):
    m -= learning_rate * (-2 / n * np.sum(x * (y - (m * x + b))))
    b -= learning_rate * (-2 / n * np.sum(y - (m * x + b)))
    c = 1 / n * np.sum(np.power(y - (m * x + b), 2))
    print(f"i: {i}, m: {m:.5f}, b: {b:.5f}, c: {c:.5f}")
