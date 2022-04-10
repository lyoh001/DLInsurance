#%%
import io
import secrets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

#%%
df, y_label = (
    pd.read_csv(
        io.StringIO(
            """age,affordibility,bought_insurance
22,1,0
25,0,0
47,1,1
52,0,0
46,1,1
56,1,1
55,0,0
60,0,1
62,1,1
61,1,1
18,1,0
28,1,0
27,0,0
29,0,0
49,1,1
55,1,1
25,0,1
58,1,1
19,0,0
18,1,0
21,1,0
26,0,0
40,1,1
45,1,1
50,1,1
54,1,1
23,1,0
46,1,0 """
        )
    ),
    "bought_insurance",
)
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25
)  # stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train
#%%
y.value_counts(normalize=True), y_train.value_counts(
    normalize=True
), y_test.value_counts(normalize=True)
#%%
pipe = make_pipeline(
    (FunctionTransformer(lambda x: x / 100)),
)
preprocessor = make_column_transformer(
    (pipe, ["age"]),
    (MinMaxScaler(), ["affordibility"]),
)

X_train = preprocessor.fit_transform(X_train)
X_train
#%%
print(X_train[0].shape, len(np.unique(y_train)))
#%%
model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(X_train[0].shape)),
        keras.layers.Dense(
            len(np.unique(y_train)) - 1,
            activation="sigmoid",
            bias_initializer="zeros",
            kernel_initializer="ones",
        ),
    ]
)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(X_train, y_train, epochs=5000)
#%%
model.evaluate(preprocessor.transform(X_test), y_test)
#%%
preprocessor.transform(X_test)
#%%
X_test
#%%
y_test
#%%
model.predict(preprocessor.transform(X_test))
#%%
coef, intercept = model.get_weights()
coef, intercept
#%%
plt.figure(figsize=(7, 7))
sns.heatmap(
    tf.math.confusion_matrix(
        y_test, np.round(model.predict(preprocessor.transform(X_test)))
    ),
    annot=True,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
#%%
n = secrets.randbelow(X_test.shape[0])
print("============================================")
print(f"Random Index: {n}")
print(f"Test value: {y_test.iloc[n]}")
print(f"Test probability:\t{model.predict(preprocessor.transform(X_test)[n:n+1])[0]}")
print("============================================")
print(f"Test values:\t\t{list(y_test.values)}")
print(
    f"Test max prob indices:\t{np.transpose(np.round(model.predict(preprocessor.transform(X_test))))}"
)
print("============================================")
