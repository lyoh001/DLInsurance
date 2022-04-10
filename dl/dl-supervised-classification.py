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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from keras.wrappers.scikit_learn import KerasClassifier
from keras_tuner.tuners import RandomSearch
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
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
df, y_label = load_iris(as_frame=True)["frame"], "target"
df, y_label = pd.read_csv("../.data/kaggle.csv"), "target"
df, y_label = pd.read_csv("../.data/titanic.csv", index_col=0), "Survived"
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
    stratify=y,
    random_state=11,
)
print(f"y.value_counts:\n{y.value_counts(normalize=True)}")
print(f"y_train.value_counts:\n{y_train.value_counts(normalize=True)}")
print(f"y_test.value_counts:\n{y_test.value_counts(normalize=True)}")
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
            units=[1, df[y_label].nunique()][df[y_label].nunique() > 2],
            activation=["sigmoid", "softmax"][df[y_label].nunique() > 2],
        )
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss=["binary_crossentropy", "categorical_crossentropy"][df[y_label].nunique() > 2],
        metrics=["accuracy"],
    )
    return model


tuner = RandomSearch(
    tune_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=3,
    overwrite=True,
    directory=".",
    project_name="keras_tuner",
)

tuner.search_space_summary()
#%%
tuner.search(X_train_processed, [y_train, keras.utils.to_categorical(y_train)][df[y_label].nunique() > 2], epochs=EPOCHS_KT, validation_data=(X_test_processed, [y_test, keras.utils.to_categorical(y_test)][df[y_label].nunique() > 2]))
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
            units=[1, df[y_label].nunique()][df[y_label].nunique() > 2],
            activation=["sigmoid", "softmax"][df[y_label].nunique() > 2],
        )
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=["binary_crossentropy", "categorical_crossentropy"][df[y_label].nunique() > 2],
        metrics=["accuracy"],
    )
    return model

#%%
tests = [
    {
        "model": make_pipeline_imb(
            (preprocessor),
            (SMOTE(random_state=11)),
            (KerasClassifier(build_fn=build_model)),
        ),
        "params": {
            "columntransformer__pipeline-1__simpleimputer__strategy": ["most_frequent"],
            "columntransformer__pipeline-1__simpleimputer__add_indicator": [False],
            "columntransformer__pipeline-2__simpleimputer__strategy": ["most_frequent"],
            "columntransformer__pipeline-2__simpleimputer__add_indicator": [False],
            "columntransformer__pipeline-2__onehotencoder__handle_unknown": ["ignore"],
            "columntransformer__pipeline-3__knnimputer__n_neighbors": range(1, 11),
            "columntransformer__pipeline-3__knnimputer__add_indicator": [False],
            "kerasclassifier__epochs": [EPOCHS_CV],
            "kerasclassifier__batch_size":  [128, 256] if tf.config.list_physical_devices("GPU") else [32, 64],
            "kerasclassifier__verbose": [1],
            "kerasclassifier__input": [[v for k, v in params.items() if "input" in k]],
            "kerasclassifier__hidden": [[v for k, v in params.items() if "hidden" in k]],
            "kerasclassifier__num_layers": [params["num_layers"]],
            "kerasclassifier__dropout": [round(params["dropout"], 2)],
            "kerasclassifier__learning_rate": [params["learning_rate"]],
        },
    },
]
for test in tests:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
    rscv = RandomizedSearchCV(
        estimator=test["model"],
        param_distributions=test["params"],
        n_jobs=-1,
        cv=cv,
        scoring="accuracy",
        n_iter=10,
        return_train_score=True,
    )
    rscv.fit(X_train, y_train)
    print("===train============================")
    print(f"train score: {rscv.best_score_ * 100:.2f}%\t{test['model'][-1]}\t{rscv.best_params_}")
    print("===params============================")
    display(pd.DataFrame(rscv.cv_results_).sort_values(by="rank_test_score"))
    print("===test============================")
    print(f"test score: {rscv.score(X_test, y_test) * 100:.2f}%")
    print("====end===========================\n")
#%%
rscv.best_estimator_
#%%
rscv.best_params_
#%%
rscv.best_estimator_.get_params()
#%%
print("===train============================")
print(f"train score: {rscv.best_score_ * 100:.2f}")
print("===test============================")
print(f"test score: {rscv.score(X_test, y_test) * 100:.2f}%")
print("====end===========================\n")
print(y_test[:10])
print(rscv.predict(X_test)[:10].reshape(-1, 1))
print(rscv.predict_proba(X_test)[:10, :])
#%%
print(classification_report(y_test, rscv.predict(X_test)))
