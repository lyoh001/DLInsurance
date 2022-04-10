#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

df, X_label, y_label = (
    pd.DataFrame(
        {
            "area": [1, 2, 3, 4, 5],
            "price": [5, 7, 9, 11, 13],
        }
    ),
    "area",
    "price",
)
df, X_label, y_label = (
    pd.DataFrame({"area": [0.5, 2.3, 2.9], "price": [1.4, 1.9, 3.2]}),
    "area",
    "price",
)
df, X_label, y_label = (
    pd.DataFrame(
        {
            "area": [
                1500,
                1700,
                1750,
                1800,
                1820,
                1920,
                1450,
                1590,
                1596,
                1623,
                1878,
                1658,
                1720,
                1985,
                2000,
                2100,
                2050,
                1990,
                1965,
                1970,
                2120,
                2200,
                2156,
                1269,
                1489,
                1785,
                1965,
                1948,
                2008,
                2079,
                2116,
                2230,
                2200,
                2220,
                2365,
                2325,
                2396,
                2489,
                2420,
                2398,
                2350,
                2375,
                2236,
                2347,
                2459,
            ],
            "price": [
                158900,
                169850,
                178950,
                178650,
                180000,
                186850,
                150000,
                149870,
                158620,
                159990,
                189680,
                168980,
                170000,
                190000,
                198510,
                200000,
                193580,
                200000,
                195180,
                198680,
                201650,
                220000,
                216510,
                138550,
                149850,
                179850,
                196280,
                195680,
                200000,
                205880,
                210000,
                220000,
                219850,
                222000,
                235680,
                239580,
                240000,
                248850,
                245590,
                240000,
                236840,
                230000,
                226260,
                220590,
                239840,
            ],
        }
    ),
    "area",
    "price",
)
df, X_label, y_label = (
    pd.DataFrame(
        {
            "area": [92, 56, 88, 70, 80, 49, 65, 35, 66, 67],
            "price": [98, 68, 81, 80, 83, 52, 66, 30, 68, 73],
        }
    ),
    "area",
    "price",
)
df.plot.scatter(x=X_label, y=y_label, marker="*")
scaler = MinMaxScaler()
X, y = scaler.fit_transform(df.drop(y_label, axis=1)), df[y_label]
#%%
class NN:
    def __init__(self) -> None:
        self.w0 = 0
        self.b = 0
        print(f"Constructed a NN with w0: {self.w0:.5f} and b: {self.b:.5f}")

    def fit(self, X, y):
        cost, accuracy, learning_rate = [], 0.00001, 0.1
        for i in range(5_000):
            y_pred = self.w0 * X + self.b

            w0_d = -2 * np.mean(X.T * (y.values - (self.w0 * X.T + self.b)))
            b_d = -2 * np.mean((y.values - (self.w0 * X.T + self.b)))

            self.w0 = self.w0 - learning_rate * w0_d
            self.b = self.b - learning_rate * b_d

            loss = np.mean(np.square(y.values - y_pred.T))
            print(
                f"epoch: {i}\nloss: {loss:.5f}\nw0: {self.w0:.5f}\t\tbias: {self.b:.5f}\nw0_slope: {w0_d:.5f}\tbias_slope: {b_d:.5f}\n-----------------------------------------------"
            )
            if abs(w0_d) < accuracy > abs(b_d):
                break

        for i in range(-10, 10):
            y_pred = i * 100_000 * X[:, 0] + self.b
            loss = np.mean(np.square(y - y_pred))
            cost.append(loss)
        plt.scatter(x=range(20), y=cost)

    def predict(self, X):
        return self.w0 * X + self.b


nn = NN()
nn.fit(X, y)
# ===Manual Learning========================================================
print(
    f"Predicted House Price: ${nn.predict(scaler.transform(np.array([[2400]]))[0][0]):_.2f}"
)
#%%
# ===Machine Learning========================================================
model = LinearRegression()
model.fit(X, y)
print(
    f"Constructed a LinearRegression with w0: {model.coef_[0]:.5f} and b: {model.intercept_:.5f}"
)
print(
    f"Predicted House Price: ${model.predict(scaler.transform(np.array([[2400]])))[0]:_.2f}"
)
#%%
# ===Deep Learning========================================================
model = keras.models.Sequential([keras.layers.Dense(units=1)])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1), loss="mean_squared_error"
)
model.fit(X, y, epochs=4_500)
#%%
print(model.predict(scaler.transform(np.array([[2400]]))))
print(model.weights)
