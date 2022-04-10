#%%
from google.colab import drive

drive.mount("/content/drive", force_remount=True)
#%%
%pip install imblearn
%pip install keras-tuner
%pip install numpy
%pip install pandas
%pip install sklearn
%pip install tensorflow
#%%
EPOCHS_KT, EPOCHS_CV = 5, 10
#%%
import dtale
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras_tuner.tuners import RandomSearch
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_diabetes
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from tensorflow.python.client import device_lib

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
set_config(display="diagram", print_changed_only=False)
!cat /proc/cpuinfo | grep "model name"
!cat /proc/meminfo | grep "MemTotal"
print(device_lib.list_local_devices())
print(tf.config.list_physical_devices("GPU"))
print(tf.test.gpu_device_name())
!nvidia-smi
#%%
df, y_label = load_diabetes(as_frame=True)["frame"], "target"
df, y_label = pd.read_csv("../.data/kaggle.csv"), "target"
df, y_label = pd.read_csv("../.data/dsr.csv"), "DSR"
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
df.isnull().sum().sum() / np.product(df.shape) * 100
#%%
df.corr().style.background_gradient(cmap="coolwarm")
#%%
dtale.show(df)
#%%
X, y = df.drop(y_label, axis=1), df[y_label]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=11,
)
col_oe = []
col_ohe = [
    col
    for col in X_train.columns
    if np.issubsctype(X_train[col].dtype, np.object0)
    and col not in col_oe
    and X_train[col].nunique() <= 10
]
col_num = [
    col for col in X_train.columns if np.issubsctype(X_train[col].dtype, np.number)
]
preprocessor_oe = make_pipeline(
    (SimpleImputer(strategy="most_frequent")),
    (OrdinalEncoder(categories=[["small", "medium", "large"], ["first", "second", "third"]])),
    (MinMaxScaler()),
)
preprocessor_ohe = make_pipeline(
    (SimpleImputer(strategy="most_frequent")),
    (OneHotEncoder(handle_unknown="ignore")),
)
preprocessor_num = make_pipeline(
    (KNNImputer()),
    (MinMaxScaler()),
)
preprocessor = make_column_transformer(
    (preprocessor_oe, col_oe),
    (preprocessor_ohe, col_ohe),
    (preprocessor_num, col_num),
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("-------------------------------------------------------")
print(f"X.shape: {X.shape}\t\tX_train.shape: {X_train.shape}\t\tX_test.shape: {X_test.shape}")
print(f"y.shape: {y.shape}\t\ty_train.shape: {y_train.shape}\t\ty_test.shape: {y_test.shape}")
print("-------------------------------------------------------")
print(f"col_oe: {col_oe}")
print(f"col_ohe: {col_ohe}")
print(f"col_num: {col_num}")
print(f"total cols for preprocessor: {len(col_oe) + len(col_ohe) + len(col_num)}")
print("-------------------------------------------------------")
print(f"X_train_processed:{X_train_processed.shape}")
print(f"X_test_processed: {X_test_processed.shape}")
print("-------------------------------------------------------")
#%%
def tune_model(hp):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=hp.Int("input_00", min_value=1, max_value=64, step=4),
            input_shape=X_train_processed.shape[1:],
        )
    )
    for i in range(hp.Int("num_layers", min_value=1, max_value=32)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"hidden_{i:02}", min_value=1, max_value=64, step=4),
                activation="relu",
            )
        )
        model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))
    model.add(
        keras.layers.Dense(
            units=1,
            activation="linear"
        )
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    return model


tuner = RandomSearch(
    tune_model,
    objective="val_mean_squared_error",
    max_trials=10,
    executions_per_trial=3,
    overwrite=True,
    directory=".",
    project_name="keras_tuner",
)

tuner.search_space_summary()
#%%
tuner.search(X_train_processed, y_train, epochs=EPOCHS_KT, validation_data=(X_test_processed, y_test))
print("done")
#%%
tuner.results_summary()
#%%
tuner.get_best_models(num_models=1)[0].summary()
#%%
params = dict(sorted(tuner.get_best_hyperparameters()[0].values.items()))
params
#%%
def build_model(input, hidden, num_layers, dropout, learning_rate):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=input[0],
            input_shape=X_train_processed.shape[1:]
        )
    )
    for i in range(num_layers):
        model.add(
            keras.layers.Dense(
                units=hidden[i],
                activation="relu",
            )
        )
        model.add(keras.layers.Dropout(dropout))
    model.add(
        keras.layers.Dense(
            units=1,
            activation="linear",
        )
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    return model

#%%
tests = [
    {
        "model": make_pipeline(
            (preprocessor),
            (KerasRegressor(build_fn=build_model)),
        ),
        "params": {
            "columntransformer__pipeline-1__simpleimputer__strategy": ["most_frequent"],
            "columntransformer__pipeline-1__simpleimputer__add_indicator": [False],
            "columntransformer__pipeline-2__simpleimputer__strategy": ["most_frequent"],
            "columntransformer__pipeline-2__simpleimputer__add_indicator": [False],
            "columntransformer__pipeline-2__onehotencoder__handle_unknown": ["ignore"],
            "columntransformer__pipeline-3__knnimputer__n_neighbors": range(1, 11),
            "columntransformer__pipeline-3__knnimputer__add_indicator": [False],
            "kerasregressor__epochs": [EPOCHS_CV],
            "kerasregressor__batch_size": [128, 256] if tf.config.list_physical_devices("GPU") else [32, 64],
            "kerasregressor__verbose": [1],
            "kerasregressor__input": [[v for k, v in params.items() if "input" in k]],
            "kerasregressor__hidden": [[v for k, v in params.items() if "hidden" in k]],
            "kerasregressor__num_layers": [params["num_layers"]],
            "kerasregressor__dropout": [round(params["dropout"], 2)],
            "kerasregressor__learning_rate": [params["learning_rate"]],
        },
    },
]
for test in tests:
    cv = KFold(n_splits=10, shuffle=True, random_state=11)
    rscv = RandomizedSearchCV(
        estimator=test["model"],
        param_distributions=test["params"],
        n_jobs=-1,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_iter=10,
        return_train_score=True,
    )
    rscv.fit(X_train, y_train)
    print("===train============================")
    print(f"train score: {np.sqrt(-rscv.best_score_)}\t{test['model'][-1]}\t{rscv.best_params_}")
    print("===params============================")
    display(pd.DataFrame(rscv.cv_results_).sort_values(by="rank_test_score"))
    print("===test============================")
    print(f"test score: {np.sqrt(-rscv.score(X_test, y_test))}")
    print("====end===========================\n")
#%%
rscv.best_estimator_
#%%
rscv.best_params_
#%%
rscv.best_estimator_.get_params()
#%%
print("===train============================")
print(f"train score: {np.sqrt(-rscv.best_score_)}")
print("===test============================")
print(f"test score: {np.sqrt(-rscv.score(X_test, y_test))}")
print("====end===========================\n")
print(y_test[:10])
print(rscv.predict(X_test)[:10].reshape(-1, 1))
print(f"Mean Squared Error: {np.sqrt(mean_squared_error(rscv.predict(X_test), y_test))}")
#%%
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(X, rscv.predict(X), color="red")
plt.scatter(X, y)
plt.title(
    "Polynomial Linear Regression using Tensorflow and Python 3.8",
    fontsize=10,
)
plt.xlabel("X")
plt.ylabel("y")
plt.show()
