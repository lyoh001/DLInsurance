#%%
import random
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import set_config
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_diabetes, load_digits, load_iris
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
    silhouette_score,
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
    "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/data/nigerian-songs.csv"
)
df, _ = load_iris(as_frame=True, return_X_y=True)
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
[[i, c] for i, c in enumerate(df.columns)]
#%%
df.quantile([0.01, 0.99])
#%%
df.isnull().sum()
#%%
df.isnull().sum().sum() / np.product(df.shape) * 100
#%%
df.corr().style.background_gradient(cmap="coolwarm")
#%%
sns.pairplot(df)
#%%
top = df[df.columns[3]].value_counts().iloc[:5]
plt.figure(figsize=(7, 7))
plt.title("Title")
plt.xticks(rotation=45)
sns.barplot(top.index, top.values)
#%%
sns.jointplot(data=df, hue=df.columns[3], x=df.columns[6], y=df.columns[7], kind="kde")
#%%
sns.FacetGrid(df, hue=df.columns[3], size=5).map(
    plt.scatter, df.columns[6], df.columns[7]
).add_legend()
#%%
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
ax = ax.flatten()
for i, (col, value) in enumerate(df[col_num].items()):
    sns.boxplot(y=col, data=df[col_num], ax=ax[i])
plt.tight_layout()
#%%
col_cat = [
    col
    for col in df.columns
    if df[col].dtype in ["object", "bool"] and df[col].nunique() < 10
]
col_num = [col for col in df.columns if df[col].dtype in ["float", "int"]]
col_cat, col_num, col_cat + col_num, len(col_cat + col_num)
#%%
wcss = [
    kmeans.inertia_
    for i in range(1, 10)
    if (kmeans := KMeans(n_clusters=i, init="k-means++").fit(df[col_cat + col_num]))
    is not None
]
plt.figure(figsize=(10, 10))
sns.lineplot(range(len(wcss)), wcss, marker="o")
plt.title("Elbow")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
#%%
df = df[col_cat + col_num]
f"Total number of column combs iterations: {len(list(combinations(df.columns, 2)))}"
#%%
k = 3
transformer_cat = make_pipeline(
    (SimpleImputer()),
    # (OrdinalEncoder(categories=[["Small", "Medium", "Large"], ["First", "Second", "Third"]])),
    (OneHotEncoder()),
)
transformer_num = make_pipeline(
    # (IterativeImputer()),
    (KNNImputer()),
    (PolynomialFeatures()),
    # (MinMaxScaler()),
    (StandardScaler()),
)
preprocessor = make_column_transformer(
    (transformer_cat, col_cat),
    (transformer_num, col_num),
)
df["target"] = make_pipeline((preprocessor), (KMeans(n_clusters=k))).fit_predict(df)
cols = combinations(df.columns[:-1], 2)
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
fig.suptitle("K-Means Clustering")
for row in ax:
    for col in row:
        c1, c2 = next(cols)
        col.set_title(f"{c1} & {c2}")
        for i in range(k):
            col.scatter(
                df.groupby("target").get_group(i)[c1],
                df.groupby("target").get_group(i)[c2],
            )
silhouette_score(df[col_cat + col_num], df["target"])
