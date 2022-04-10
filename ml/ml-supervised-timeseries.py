#%%
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from tensorflow import keras
from xgboost import XGBClassifier, XGBRegressor

set_config(display="diagram", print_changed_only=False)
#%%
df = pd.read_csv(
    "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/7-TimeSeries/data/energy.csv",
    date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
    parse_dates=["timestamp"],
    index_col=0,
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
df.isnull().sum().sum() / np.product(df.shape) * 100
#%%
df.corr().style.background_gradient(cmap="coolwarm")
#%%
col_cat = [
    col
    for col in df.columns
    if df[col].dtype in ["object", "bool"] and df[col].nunique() < 10
]
col_num = [
    col for col in df.columns if df[col].dtype in ["float", "int"] and col == "load"
]
col_cat, col_num, col_cat + col_num
#%%
font_size = 15
df.plot(y="load", subplots=True, figsize=(20, 10), fontsize=font_size)
plt.xlabel("Time", fontsize=font_size)
plt.ylabel("Load", fontsize=font_size)
plt.show()
#%%
print(df.index.min(), df.index.max(), df.shape)
date_split = "2014-12-01"
df_train = df[df.index < date_split][col_cat + col_num]
df_test = df[df.index >= date_split][col_cat + col_num]
df_train.shape, df_test.shape
#%%
font_size = 15
df_train.join(df_test, how="outer", lsuffix="_train", rsuffix="_test").plot(
    subplots=False, figsize=(20, 10), fontsize=font_size
)
plt.xlabel("Time", fontsize=font_size)
plt.ylabel("Load", fontsize=font_size)
plt.show()
#%%
transformer_cat = make_pipeline(
    (SimpleImputer()),
    # (OrdinalEncoder(categories=[["Small", "Medium", "Large"], ["First", "Second", "Third"]])),
    (OneHotEncoder()),
)
transformer_num = make_pipeline(
    # (IterativeImputer()),
    (KNNImputer()),
    # (PolynomialFeatures()),
    (MinMaxScaler()),
    # (StandardScaler()),
    (
        FunctionTransformer(
            lambda d: np.array(
                [
                    n
                    for n in zip(
                        d[:, 0], d[:, 0][1:], d[:, 0][2:], d[:, 0][3:], d[:, 0][4:]
                    )
                ]
            )
        )
    ),
)
preprocessor = make_column_transformer(
    (transformer_cat, col_cat),
    (transformer_num, col_num),
)
preprocessed_train_data = preprocessor.fit_transform(df_train)
preprocessed_test_data = preprocessor.transform(df_test)
preprocessed_train_data.shape, preprocessed_test_data.shape
#%%
X_train, y_train = preprocessed_train_data[:, :-1], preprocessed_train_data[:, -1]
X_test, y_test = preprocessed_test_data[:, :-1], preprocessed_test_data[:, -1]
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#%%
tests = [
    {
        "model": make_pipeline(
            (SVR()),
        ),
        "params": {
            "svr__kernel": ["rbf"],
            "svr__gamma": [0.5],
            "svr__C": [10],
            "svr__epsilon": [0.05],
        },
    },
]
for test in tests:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    rscv = RandomizedSearchCV(
        estimator=test["model"],
        param_distributions=test["params"],
        n_jobs=-1,
        # cv=cv,
        scoring="r2",
        # scoring="neg_mean_squared_error",
        n_iter=10,
        return_train_score=True,
    )
    rscv.fit(X_train, y_train)
    print("===train============================")
    print(f"{rscv.best_score_ * 100:.2f}%\t{test['model'][0]}\t{rscv.best_params_}")
    print("===params============================")
    display(pd.DataFrame(rscv.cv_results_).sort_values(by="rank_test_score"))
    print("===test============================")
    print(f"test score:{rscv.score(X_test, y_test) * 100:.2f}%")
    print("====end===========================\n")
#%%
rscv.best_estimator_
#%%
plt.figure(figsize=(20, 10))
plt.plot(
    df_test.index[:-4],
    preprocessor.named_transformers_["pipeline-2"]
    .named_steps["minmaxscaler"]
    .inverse_transform(y_test.reshape(-1, 1))[: np.newaxis],
    color="blue",
)
plt.plot(
    df_test.index[:-4],
    preprocessor.named_transformers_["pipeline-2"]
    .named_steps["minmaxscaler"]
    .inverse_transform(rscv.predict(X_test).reshape(-1, 1))[: np.newaxis],
    color="red",
)
plt.legend(["Actual", "Predicted"])
plt.xlabel("Timestamp")
plt.title("Test data prediction")
plt.show()
#%%
plt.figure(figsize=(20, 10))
plt.plot(
    df_train.index[:-4],
    preprocessor.named_transformers_["pipeline-2"]
    .named_steps["minmaxscaler"]
    .inverse_transform(y_train.reshape(-1, 1))[: np.newaxis],
    color="blue",
)
plt.plot(
    df_train.index[:-4],
    preprocessor.named_transformers_["pipeline-2"]
    .named_steps["minmaxscaler"]
    .inverse_transform(rscv.predict(X_train).reshape(-1, 1))[: np.newaxis],
    color="red",
)
plt.legend(["Actual", "Predicted"])
plt.xlabel("Timestamp")
plt.title("Training data prediction")
plt.show()
#%%
y_test[:10]
#%%
preprocessor.named_transformers_["pipeline-2"].named_steps[
    "minmaxscaler"
].inverse_transform(y_test.reshape(-1, 1))[: np.newaxis][:10]
#%%
rscv.predict(X_test)[:10]
#%%
preprocessor.named_transformers_["pipeline-2"].named_steps[
    "minmaxscaler"
].inverse_transform(rscv.predict(X_test).reshape(-1, 1))[: np.newaxis][:10]
#%%
rscv.best_estimator_.get_params()
