#%%
import numpy as np

y = np.array([0.3, 0.7, 1, 0, 0.5])
y_pred = np.array([1, 1, 0, 0, 1])


def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))


def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))


def bce(y, y_pred):
    y_pred_ref = np.array([n + (1e-15 * (-1 if n else 1)) for n in y_pred])
    return -np.mean(y * np.log(y_pred_ref) + (1 - y) * np.log(1 - y_pred_ref))


print(mae(y, y_pred))
print(mse(y, y_pred))
print(bce(y, y_pred))
