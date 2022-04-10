#%%
import os
import shutil

path_source, path_dest = "/home/lyoh001/Downloads", "/home/lyoh001/vscode/AzureDL/data"
[
    shutil.move(
        os.path.join(path_source, f),
        os.path.join(path_dest, "data.csv"),
    )
    for f in os.listdir(path_source)
    if ".csv" in f
]
os.chdir(path_dest)
os.system("git add .")
os.system("git commit -m 'Commit'")
os.system("git push")
#%%
%pip install dabl
# %pip install dtale
# %pip install imblearn
# %pip install keras-tuner
# %pip install numpy
# %pip install pandas
# %pip install sklearn
# %pip install tensorflow
#%%
import datetime
import shutil
import warnings
from pickle import dump

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from dabl import SimpleClassifier, SimpleRegressor, plot
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from IPython.display import display
from keras_tuner.tuners import BayesianOptimization, Hyperband, RandomSearch
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import (load_breast_cancer, load_diabetes, load_iris,
                              load_wine)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import device_lib

%matplotlib inline
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
set_config(display="diagram", print_changed_only=False)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
plt.rcParams["figure.figsize"] = [12.8, 7.2]
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")
print(device_lib.list_local_devices())
print(tf.config.list_physical_devices("GPU"))
print(tf.test.gpu_device_name())
!cat /proc/cpuinfo | grep "model name"
!cat /proc/meminfo | grep "MemTotal"
!nvidia-smi
#%%
RANDOM_STATE = 11
SEARCH = ["hyperband", "random", "bayesian"][0]
EPOCHS = 500
MAX_TRIALS = 20
DUPLICATES = 0
CLASSIFICATION = 0
#%%
df, y_label = load_breast_cancer(as_frame=True)["frame"], "target"
df, y_label = load_iris(as_frame=True)["frame"], "target"
df, y_label = load_wine(as_frame=True)["frame"], "target"
df, y_label = load_diabetes(as_frame=True)["frame"], "target"
df, y_label = pd.read_csv("https://raw.githubusercontent.com/lyoh001/AzureDL/main/data/boston.csv", delimiter=","), "MEDV"
df, y_label = pd.read_csv("https://raw.githubusercontent.com/lyoh001/AzureDL/main/data/titanic.csv", delimiter=","), "Survived"
df, y_label = pd.read_csv("https://raw.githubusercontent.com/lyoh001/AzureDL/main/data/data.csv", delimiter=","), "target"

print(f"Current Shape: {df.shape}.")
print("-------------------------------------------------------")
print(f"Duplicates Percentage: {df.duplicated().sum() / df.shape[0] * 100:.2f}%")
if DUPLICATES:
    print(f"Duplicates have been kept {df.shape}.")
else:
    df.drop_duplicates(inplace=True)
    print(f"Duplicates have been removed {df.shape}.")
display(df.sample(3))
#%%
# df[""] = pd.to_datetime(df[""], format="%d/%m/%Y %H:%M:%S")
# df["year"] = df[""].dt.year
# df["month"] = df[""].dt.month
# df["dayofweek"] = df[""].dt.dayofweek
# df["dates"] = (df[""] - df[""]).dt.days
# df.replace({"A": 0, "B": 1, "unknown": np.nan}, inplace=True)
# df[""] = df[""].map(lambda x: {"": 0, "": 1}.get(x, np.nan))
# for col in [""]:
#     print(f"col: {col}")
#     display(df[col][~df[col].map(lambda x: isinstance(x, (int, float)))])
#     df[col] = df[col].str.strip()
#     df[col] = df[col].map(pd.to_numeric)
#     df[col] = df[col].astype(float)
# df.drop([""], inplace=True, axis=1)
df.dropna(subset=[y_label], inplace=True)
print("Data cleaning has been completed.")
#%%
print(f"Current Shape: {df.shape}.")
df_info = pd.DataFrame(
    {
        "column": [col for col in df.columns],
        "dtype": [f"{df[col].dtype}" for col in df.columns],
        "na": [f"{df[col].isna().sum()}" for col in df.columns],
        "na %": [f"{round(df[col].isna().sum() / df[col].shape[0] * 100)}%" for col in df.columns],
        "outliers": [f"{((df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))).sum()}" if np.issubsctype(df[col].dtype, np.number) else "n/a" for col in df.columns],
        "outliers %": [f"{round((((df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))).sum()) / df[col].shape[0] * 100)}%" if np.issubsctype(df[col].dtype, np.number) else "n/a" for col in df.columns],
        "skewness": [f"{df[col].skew(axis=0, skipna=True):.2f}" if np.issubsctype(df[col].dtype, np.number) else "n/a" for col in df.columns],
        "corr": [f"{round(df[col].corr(other=df[y_label]) * 100)}%" if np.issubsctype(df[col].dtype, np.number) else "n/a" for col in df.columns],
        "nunique": [f"{df[col].nunique()}" for col in df.columns],
        "unique": [df[col].unique() for col in df.columns],
    }
).sort_values(by="dtype", ascending=False)
display(df_info)
#%%
OUTLIERS = ["keep", "cap", "log_transform", "drop"][0]
col_outlier = [col for col in df.columns if np.issubsctype(df[col].dtype, np.number) and col in [""]]
q1, q3 = df[col_outlier].quantile(0.25), df[col_outlier].quantile(0.75)
iqr = q3 - q1
lower_range, upper_range = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
condition = ~((df[col_outlier] < lower_range) | (df[col_outlier] > upper_range)).any(axis=1)
print(f"Current Shape: {df.shape}.")
print("-------------------------------------------------------")
print(f"Scanning for outliers in {col_outlier}.")
print(f"Outliers Percentage: {(df.shape[0] - df[condition].shape[0]) / df.shape[0] * 100:.2f}%")
if OUTLIERS == "keep":
    print(f"Outliers have been kept {df.shape}.")
elif OUTLIERS == "cap":
    for col in col_outlier:
        df[col] = np.where(df[col] < lower_range[col], lower_range[col], df[col])
        df[col] = np.where(df[col] > upper_range[col], upper_range[col], df[col])
    print(f"Outliers have been capped {df.shape}.")
elif OUTLIERS == "log_transform":
    for col in col_outlier:
        df[col] = np.log(df[col])
    print(f"Outliers have been log transformed {df.shape}.")
else:
    df = df[condition]
    print(f"Outliers have been removed {df.shape}.")
#%%
plt.title("Boxplots for Numeric Columns")
sns.boxplot(
    data=df[[col for col in df.columns if np.issubsctype(df[col].dtype, np.number)]],
    orient="h",
    color="steelblue"
)
plt.grid()
plt.show()
#%%
sns.heatmap(df.corr(), cmap="Blues", fmt=".2f", annot=True, linewidths=1)
plt.show()
#%%
for col in df.columns:
    if np.issubsctype(df[col].dtype, np.number):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        sns.set(style="white", palette="muted", color_codes=True)
        sns.distplot(x=df[col], ax=ax[0], color="steelblue", kde=True).set_xlabel(f"{col}")
        sns.boxplot(x=df[col], ax=ax[1], color="steelblue").set_xlabel(f"{col}")
plt.show()
#%%
if CLASSIFICATION:
    for col in df.columns:
        if np.issubsctype(df[col].dtype, np.number):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            sns.set(style="white", palette="muted", color_codes=True)
            sns.boxplot(x=y_label, y=col, data=df, color="steelblue")
    plt.show()
#%%
display(df.describe().round(2).T.style.background_gradient(cmap="Blues"))
display(df.quantile([0.01, 0.99]).T.style.background_gradient(cmap="Blues"))
#%%
OVERSAMPLE = ["none", "undersample", "oversample", "combine"][0]
X, y = df.drop(y_label, axis=1), df[y_label]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=[None, y, y][CLASSIFICATION],
    random_state=RANDOM_STATE,
)
col_oe = []
preprocessor_oe = make_pipeline(
    (SimpleImputer(strategy="most_frequent")),
    (OrdinalEncoder(categories=[["no", "yes"]])),
    (MinMaxScaler()),
)
col_ohe = [
    col
    for col in X_train.columns
    if np.issubsctype(X_train[col].dtype, np.object0)
    and col not in col_oe
    and X_train[col].nunique() <= 10
]
preprocessor_ohe = make_pipeline(
    (SimpleImputer(strategy="most_frequent")),
    (OneHotEncoder(drop="first", handle_unknown="ignore")),
)
col_num = [
    col for col in X_train.columns if np.issubsctype(X_train[col].dtype, np.number)
]
preprocessor_num = make_pipeline(
    (KNNImputer()),
    (MinMaxScaler()),
)
preprocessor = make_column_transformer(
    (preprocessor_oe, col_oe),
    (preprocessor_ohe, col_ohe),
    (preprocessor_num, col_num),
    sparse_threshold=0
)
X_train_processed, y_train_processed = preprocessor.fit_transform(X_train), y_train
X_test_processed, y_test_processed = preprocessor.transform(X_test), y_test
print("-------------------------------------------------------")
print(f"total na %: {df.isnull().sum().sum() / np.product(df.shape) * 100:.2f}%")
print("-------------------------------------------------------")
print(f"col_oe: {col_oe}")
print(f"col_ohe: {col_ohe}")
print(f"col_num: {col_num}")
print(f"total cols for preprocessor: {len(col_oe) + len(col_ohe) + len(col_num)}")
if CLASSIFICATION:
    if OVERSAMPLE == "undersample":
        X_train_processed, y_train_processed = RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy="not minority").fit_resample(X_train_processed, y_train_processed)
    elif OVERSAMPLE == "oversample":
        X_train_processed, y_train_processed = SMOTE(random_state=RANDOM_STATE, sampling_strategy="not majority").fit_resample(X_train_processed, y_train_processed)
    elif OVERSAMPLE == "combine":
        X_train_processed, y_train_processed = SMOTEENN(random_state=RANDOM_STATE, sampling_strategy="not majority").fit_resample(X_train_processed, y_train_processed)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    sns.set(style="white", palette="muted", color_codes=True)
    sns.despine(left=True)
    sns.countplot(y, ax=ax[0], palette="Blues").set_xlabel("y")
    sns.countplot(y_train, ax=ax[1], palette="Blues").set_xlabel("y_train")
    sns.countplot(y_train_processed, ax=ax[2], palette="Blues").set_xlabel("y_train_processed")
    plt.show()
    print("-------------------------------------------------------")
    print(f"y:\n{y.value_counts(normalize=True)}")
    print("-------------------------------------------------------")
    print(f"y_train:\n{y_train.value_counts(normalize=True)}")
    print("-------------------------------------------------------")
    print(f"y_train_processed:\n{y_train_processed.value_counts(normalize=True)}")
    print("-------------------------------------------------------")
    print(f"y_test:\n{y_test.value_counts(normalize=True)}")
    print("-------------------------------------------------------")
    print(f"y_test_processed:\n{y_test_processed.value_counts(normalize=True)}")
print("-------------------------------------------------------")
print(f"X: {X.shape}\tX_train: {X_train.shape}\tX_train_processed:{X_train_processed.shape}\tX_test: {X_test.shape}\t\tX_test_processed:{X_test_processed.shape}")
print(f"y: {y.shape}\ty_train: {y_train.shape}\t\ty_train_processed:{y_train_processed.shape}\ty_test: {y_test.shape}\t\ty_test_processed:{y_test_processed.shape}")
print("-------------------------------------------------------")
#%%
for col in col_oe + col_ohe:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.set(style="white", palette="muted", color_codes=True)
    sns.despine(left=True)
    sns.countplot(x=df[col], ax=ax[0], color="steelblue", hue=df[y_label] if CLASSIFICATION else None).set_xlabel(f"{col}")
    ax[1].pie(x=df[col].value_counts(), colors=sns.color_palette("Blues"), autopct="%.1f%%", shadow=True, labels=df[col].value_counts().index)
    ax[1].set_title(col)
plt.show()
#%%
if len(col_oe + col_ohe) >= 1 and len(col_num) >= 2:
    sns.lineplot(data=df, x=col_num[0], y=col_num[1], hue=(col_oe + col_ohe)[0])
    plt.title(f"{col_num[0].capitalize()} index with {col_num[1].capitalize()}")
    plt.show()
#%%
sns.pairplot(df, hue=y_label if CLASSIFICATION else None)
#%%
# import dtale
# dtale.show(df)
plot(X[col_num], y)
#%%
def build_ml_model():
    tests = [
        {
            "model": make_pipeline(
                (preprocessor),
                (SelectPercentile()),
                (RandomForestClassifier()) if CLASSIFICATION else (LinearRegression()),
            ),
            "params": {
                "columntransformer__pipeline-3__knnimputer__n_neighbors": [1, 3, 5, 7, 9],
                "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
                "selectpercentile__score_func": [chi2, f_classif],
                "randomforestclassifier__n_estimators": [100, 150, 200, 500],
                "randomforestclassifier__criterion": ["gini", "entropy"],
                "randomforestclassifier__max_depth": [5, 10, 20, 50, 100, 200],
                "randomforestclassifier__min_samples_split": [2, 5, 10, 20, 50, 100, 200],
                "randomforestclassifier__min_samples_leaf": [5, 10, 20, 50, 100, 200],
                "randomforestclassifier__max_features": ["auto", "sqrt", "log2"],
            }
            if CLASSIFICATION
            else {
                "columntransformer__pipeline-3__knnimputer__n_neighbors": [1, 3, 5, 7, 9],
                "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
                "selectpercentile__score_func": [chi2, f_classif],
            },
        },
    ]
    for test in tests:
        rscv = RandomizedSearchCV(
            estimator=test["model"],
            param_distributions=test["params"],
            n_jobs=-1,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
            if CLASSIFICATION
            else 10,
            scoring="accuracy" if CLASSIFICATION else "r2",
            n_iter=10,
            return_train_score=True,
        )
        rscv.fit(X_train, y_train)
        print("===train============================")
        print(f"{rscv.best_score_ * 100:.2f}%\n{test['model'][-1]}\n{rscv.best_params_}")
        print("===params============================")
        display(pd.DataFrame(rscv.cv_results_).sort_values(by="rank_test_score"))
        print("===test============================")
        print(f"test score:{rscv.score(X_test, y_test) * 100:.2f}%")
        print("====end===========================\n")

    if CLASSIFICATION:
        SimpleClassifier(random_state=RANDOM_STATE).fit(df, target_col=y_label)
        print("-------------------------------------------------------")
        print(
            classification_report(
                y_test,
                rscv.predict(X_test),
            )
        )
        sns.heatmap(
            tf.math.confusion_matrix(
                y_test,
                rscv.predict(X_test),
            ),
            cmap="Blues",
            fmt="d",
            annot=True,
            linewidths=1,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

    else:
        SimpleRegressor(random_state=RANDOM_STATE).fit(df, target_col=y_label)
        print("-------------------------------------------------------")
        print(
            f"r2: {r2_score(y_test, rscv.predict(X_test)):.3f} neg_mean_squared_error: -{mean_squared_error(y_test, rscv.predict(X_test)):_.3f}"
        )

        plt.subplot(1, 3, 1)
        sns.regplot(y_train, y_train, color="darkorange", label="Truth")
        sns.regplot(
            y_test,
            rscv.predict(X_test),
            color="darkcyan",
            label="Predicted",
        )
        plt.title(
            "Truth vs Predicted",
            fontsize=10,
        )
        plt.xlabel("Truth values")
        plt.ylabel("Predicted values")
        plt.legend()
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.scatter(
            rscv.predict(X_train),
            rscv.predict(X_train) - y_train,
            c="darkorange",
            marker="o",
            s=35,
            alpha=0.5,
            label="Train data",
        )
        plt.scatter(
            rscv.predict(X_test),
            rscv.predict(X_test) - y_test,
            c="darkcyan",
            marker="o",
            s=35,
            alpha=0.7,
            label="Test data",
        )
        plt.title(
            "Predicted vs Residuals",
            fontsize=10,
        )
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.legend(loc="upper right")
        plt.hlines(y=0, xmin=0, xmax=df[y_label].max(), lw=2, color="red")
        plt.grid()

        plt.subplot(1, 3, 3)
        sns.distplot((y_train - rscv.predict(X_train)))
        plt.title("Error Terms")
        plt.xlabel("Errors")
        plt.grid()

    plt.show()
    display(
        pd.DataFrame(
            {
                "Truth": y_test[:10].values,
                "Predicted": rscv.predict(X_test[:10]).round(1),
            }
        )
    )

def build_dl_model(hp):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=hp.Int("input_00", min_value=32, max_value=512, step=32),
            input_shape=X_train_processed.shape[1:],
        )
    )
    for i in range(1, hp.Int("num_layers", min_value=2, max_value=64)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"hidden_{i:02}", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(keras.layers.Dropout(hp.Float("dropout", min_value=0, max_value=0.5, step=0.1)))
    model.add(
        keras.layers.Dense(
            units=[1, 1, df[y_label].nunique()][CLASSIFICATION],
            activation=["linear", "sigmoid", "softmax"][CLASSIFICATION],
        )
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss=["mean_squared_error", "binary_crossentropy", "sparse_categorical_crossentropy"][CLASSIFICATION],
        metrics=["mean_squared_error", "accuracy", "accuracy"][CLASSIFICATION],
    )
    return model

def get_result(epochs):
    model = tuner.hypermodel.build(best_hps)
    model.fit(
        X_train_processed,
        y_train_processed,
        batch_size=256 if tf.config.list_physical_devices("GPU") else 64,
        epochs=epochs,
        validation_split=0.2,
        verbose=1,
    )
    if CLASSIFICATION:
        SimpleClassifier(random_state=RANDOM_STATE).fit(df, target_col=y_label)
        print("-------------------------------------------------------")
        print(
            classification_report(
                y_test_processed,
                [
                    model.predict(X_test_processed).round(),
                    np.argmax(model.predict(X_test_processed), axis=1),
                ][CLASSIFICATION - 1],
            )
        )
        sns.heatmap(
            tf.math.confusion_matrix(
                y_test_processed,
                [
                    model.predict(X_test_processed).round(),
                    np.argmax(model.predict(X_test_processed), axis=1),
                ][CLASSIFICATION - 1],
            ),
            cmap="Blues",
            fmt="d",
            annot=True,
            linewidths=1,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

    else:
        SimpleRegressor(random_state=RANDOM_STATE).fit(df, target_col=y_label)
        print("-------------------------------------------------------")
        print(f"r2: {r2_score(y_test_processed, model.predict(X_test_processed).T[0]):.3f} neg_mean_squared_error: -{mean_squared_error(y_test_processed, model.predict(X_test_processed)):_.3f}")
    
        plt.subplot(1, 3, 1)
        sns.regplot(y_train_processed, y_train_processed, color="darkorange", label="Truth")
        sns.regplot(
            y_test_processed,
            model.predict(X_test_processed).T[0],
            color="darkcyan",
            label="Predicted",
        )
        plt.title(
            "Truth vs Predicted",
            fontsize=10,
        )
        plt.xlabel("Truth values")
        plt.ylabel("Predicted values")
        plt.legend()
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.scatter(
            model.predict(X_train_processed).T[0],
            model.predict(X_train_processed).T[0] - y_train_processed,
            c="darkorange",
            marker="o",
            s=35,
            alpha=0.5,
            label="Train data",
        )
        plt.scatter(
            model.predict(X_test_processed).T[0],
            model.predict(X_test_processed).T[0] - y_test_processed,
            c="darkcyan",
            marker="o",
            s=35,
            alpha=0.7,
            label="Test data",
        )
        plt.title(
            "Predicted vs Residuals",
            fontsize=10,
        )
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.legend(loc="upper right")
        plt.hlines(y=0, xmin=0, xmax=df[y_label].max(), lw=2, color="red")
        plt.grid()

        plt.subplot(1, 3, 3)
        sns.distplot((y_train_processed - model.predict(X_train_processed).T[0]))
        plt.title("Error Terms")
        plt.xlabel("Errors")
        plt.grid()

    plt.show()

    display(
        pd.DataFrame(
            {
                "Truth": y_test_processed[:10].values,
                "Predicted": [
                    model.predict(X_test_processed[:10]).T[0],
                    model.predict(X_test_processed[:10]).T[0].round(),
                    np.argmax(model.predict(X_test_processed[:10]), axis=1),
                ][CLASSIFICATION],
            }
        )
    )
    return model

if SEARCH == "hyperband":
    tuner = Hyperband(
        build_dl_model,
        objective=["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION],
        max_epochs=MAX_TRIALS,
        factor=3,
        directory=".",
        project_name="keras_tuner",
        overwrite=True,
    )
elif SEARCH == "random":
    tuner = RandomSearch(
        build_dl_model,
        objective=["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION],
        max_trials=MAX_TRIALS,
        executions_per_trial=3,
        directory=".",
        project_name="keras_tuner",
        overwrite=True,
    )
else:
    tuner = BayesianOptimization(
        build_dl_model,
        objective=["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION],
        max_trials=MAX_TRIALS,
        executions_per_trial=3,
        directory=".",
        project_name="keras_tuner",
        overwrite=True,
    )
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(MAX_TRIALS/4))
tuner.search_space_summary()
#%%
%%time
tuner.search(
    X_train_processed,
    y_train_processed,
    batch_size=256 if tf.config.list_physical_devices("GPU") else 64,
    callbacks=[early_stop],
    epochs=MAX_TRIALS,
    validation_split=0.2,
    verbose=1,
)
tuner.results_summary()
#%%
%%time
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train_processed,
    y_train_processed,
    batch_size=256 if tf.config.list_physical_devices("GPU") else 64,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=1,
)
val_per_epoch = history.history[
    ["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION]
]
best_epoch = val_per_epoch.index([min(val_per_epoch), max(val_per_epoch), max(val_per_epoch)][CLASSIFICATION]) + 1

plt.subplot(1, 2, 1)
plt.plot(history.history[["mean_squared_error", "accuracy", "accuracy"][CLASSIFICATION]], color='deeppink', linewidth=2.5)
plt.plot(history.history[["val_mean_squared_error", "val_accuracy", "val_accuracy"][CLASSIFICATION]], color='darkturquoise', linewidth=2.5)
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy", "Val Accuracy"], loc="lower right")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], color='deeppink', linewidth=2.5)
plt.plot(history.history["val_loss"], color='darkturquoise', linewidth=2.5)
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training Loss", "Val Loss"], loc="upper right")
plt.grid()
plt.show()
#%%
%%time
print(f"Best epoch: {best_epoch}")
model = get_result(best_epoch)
#%%
# model = get_result(100)
#%%
build_ml_model()
#%%
model.summary()
plot_model(model, show_shapes=True)
#%%
model.save(f"dl_model_{time_stamp}")
shutil.make_archive(f"dl_model_{time_stamp}", "zip", f"./dl_model_{time_stamp}")
dump(preprocessor, open(f"dl_preprocessor.pkl", "wb"))
#%%
# import shutil
# from pickle import load

# import pandas as pd
# from tensorflow import keras

# df = pd.DataFrame(
#     {
#         "": [],
#         "": [],
#     }
# )
# shutil.unpack_archive("dl_model.zip", "dl_model")
# preprocessor = load(open("dl_preprocessor.pkl", "rb"))
# model = keras.models.load_model("dl_model")
# model.predict(preprocessor.transform(df))
