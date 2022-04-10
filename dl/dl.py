#%%
import secrets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras

#%%
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train[0].shape, len(np.unique(y_train)))
#%%
plt.figure(figsize=(20, 20))
n = secrets.randbelow(X_train.shape[0]) - 10
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_train[n + i], cmap=plt.get_cmap("gray"))
    print(y_train[n + i], end=" ")
#%%
print(X_train[0].shape, len(np.unique(y_train)))
#%%
model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(X_train[0].shape)),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(
            len(np.unique(y_train)),
            # activation="sigmoid",
            activation="softmax",
            bias_initializer="zeros",
            kernel_initializer="ones",
        ),
    ]
)
optimizer = "adam"
tb_callbacks = keras.callbacks.TensorBoard(
    log_dir=f"../.dl/logs/{optimizer}", histogram_freq=1, write_graph=True, write_images=True
)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(X_train, y_train, epochs=10, callbacks=[tb_callbacks])
#%%
model.evaluate(X_test, y_test)
#%%
model.summary()
#%%
pd.DataFrame(model.history.history).plot()
#%%
%matplotlib inline
%load_ext tensorboard
%tensorboard --logdir=../.dl/logs
#%%
plt.figure(figsize=(7, 7))
sns.heatmap(
    tf.math.confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=1)),
    annot=True,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
#%%
n = secrets.randbelow(X_test.shape[0])
plt.imshow(X_test[n], cmap=plt.get_cmap("gray"))
print("============================================")
print(f"Random Index: {n}")
print(f"Test value: {y_test[n]}")
print(f"Test max prob index: {np.argmax(model.predict(X_test[n:n+1])[0])}")
print(f"Test probability:\n{model.predict(X_test[n:n+1])[0]}")
print("============================================")
print(f"Test values:\t\t{y_test[n-5:n]}")
print(f"Test max prob indices:\t{np.argmax(model.predict(X_test[n-5:n]), axis=1)}")
print("============================================")
