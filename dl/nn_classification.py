#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras as keras
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

#%%
df, y_label = pd.read_csv("../.data/titanic.csv", index_col=0), "Survived"
df, y_label = load_breast_cancer(as_frame=True)["data"], "target"
df["target"] = load_breast_cancer(as_frame=True)["target"]
df
#%%
df.describe()
#%%
df.info()
#%%
df.shape
#%%
df.head(3)
#%%
df.columns
#%%
df.quantile([0.01, 0.99])
#%%
df.isnull().sum()
#%%
df.corr().style.background_gradient(cmap="coolwarm")
#%%
sns.pairplot(df)
#%%
X, y = df.drop(y_label, axis=1), df[y_label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#%%
y.value_counts(normalize=True), y_train.value_counts(
    normalize=True
), y_test.value_counts(normalize=True)
#%%
col_cat = [
    col
    for col in X_train.columns
    if X_train[col].dtype in ["object", "bool"] and X_train[col].nunique() < 10
]
col_num = [col for col in X_train.columns if X_train[col].dtype in ["float", "int"]]
col_cat, col_num = [], [
    "worst perimeter",
    "worst concave points",
]
col_cat, col_num = [], ["Age", "Pclass"]
col_cat, col_num, col_cat + col_num
#%%
transformer_cat = make_pipeline(
    (SimpleImputer(strategy="most_frequent")),
    (OneHotEncoder(handle_unknown="ignore")),
)
transformer_num = make_pipeline(
    (KNNImputer()),
    (MinMaxScaler()),
)
preprocessor = make_column_transformer(
    (transformer_cat, col_cat),
    (transformer_num, col_num),
)
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
X_train_scaled.shape, y_train.shape, X_test_scaled.shape, y_test.shape
#%%
class NN:
    def __init__(self) -> None:
        self.w0 = 0
        self.w1 = 0
        self.b = 0
        print(
            f"Constructed a NN with w0: {self.w0:.5f}, with w1: {self.w1:.5f} and bias: {self.b:.5f}"
        )

    def fit(self, X, y):
        cost, accuracy, learning_rate = [], 0.0001, 0.9
        for i in range(50_000):
            z = self.w0 * X[:, 0] + self.w1 * X[:, 1] + self.b
            y_pred = 1 / (1 + np.exp(-z))

            w0_d = np.mean(X[:, 0] * (y_pred - y.values))
            w1_d = np.mean(X[:, 1] * (y_pred - y.values))
            b_d = np.mean(y_pred - y.values)

            self.w0 = self.w0 - learning_rate * w0_d
            self.w1 = self.w1 - learning_rate * w1_d
            self.b = self.b - learning_rate * b_d

            loss = -np.mean(
                y.values * np.log(y_pred) + (1 - y.values) * np.log(1 - y_pred)
            )
            print(
                f"epoch: {i}\nloss: {loss:.5f}\nw0: {self.w0:.5f}\t\tw1: {self.w1:.5f}\t\tbias: {self.b:.5f}\nw0_slope: {w0_d:.5f}\tw1_slope: {w1_d:.5f}\tbias_slope: {b_d:.5f}\n-----------------------------------------------"
            )
            if abs(w0_d) < accuracy and abs(w1_d) < accuracy and abs(b_d) < accuracy:
                break

        for i in range(-200, 200):
            z = i * X[:, 0] + self.b
            y_pred = 1 / (1 + np.exp(-z))
            loss = -np.mean(
                y.values * np.log(y_pred) + (1 - y.values) * np.log(1 - y_pred)
            )
            # loss = np.mean(np.square(y - y_pred))
            cost.append(loss)
        plt.scatter(x=range(400), y=cost)

    def predict(self, X):
        z = self.w0 * X[:, 0] + self.w1 * X[:, 1] + self.b
        return 1 / (1 + np.exp(-z))


nn = NN()
nn.fit(X_train_scaled, y_train)
#%%
# ===Manual Learning========================================================
nn.predict(X_test_scaled)[-10:], y_test[-10:]
#%%
# ===Machine Learning========================================================
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print(model.predict_proba(X_test_scaled))
print(model.score(X_test_scaled, y_test))
print(
    f"Constructed a LogisticRegression with w0: {model.coef_[0][0]:.5f}, w1: {model.coef_[0][1]:.5f} and bias: {model.intercept_[0]:.5f}"
)
#%%
# ===Deep Learning========================================================
model = keras.models.Sequential(
    [
        keras.layers.Dense(
            input_dim=X_train_scaled.shape[1], units=1, activation="sigmoid"
        ),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_scaled, y_train, epochs=4_000)
#%%
print(model.predict(X_test_scaled)[-10:])
print(model.weights)
