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
df, y_label = (
    load_diabetes(as_frame=True)["frame"].iloc[:, [2, -1]].sort_values(by=["bmi"]),
    "target",
)
df, y_label = (
    pd.DataFrame(
        {
            "X": [i for i in range(100)],
            "target": [random.gauss(i, 30) for i in range(100)],
        },
    ),
    "target",
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
sns.pairplot(df)
#%%
sns.catplot(x=y_label, y=df.columns[4], data=df, kind="swarm")
#%%
sns.boxplot(x=y_label, y=df.columns[4], data=df)
#%%
sns.displot(df[y_label], kind="kde")
#%%
h = sns.FacetGrid(df, row=df.columns[3], col=df.columns[1], hue=y_label)
h.map(plt.hist, df.columns[2], alpha=0.75)
h.add_legend()
#%%
oe = OrdinalEncoder()
scaler = MinMaxScaler()
pd.Series(
    mutual_info_classif(
        scaler.fit_transform(oe.fit_transform(df.dropna().drop(y_label, axis=1))),
        df.dropna()[y_label],
    ),
    index=X_train.columns,
).sort_values(ascending=False)
#%%
X, y = df.drop(y_label, axis=1), df[y_label]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=11,
)
X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape
#%%
y.value_counts(normalize=True), y_train.value_counts(
    normalize=True
), y_test.value_counts(normalize=True),
#%%
col_cat = [
    col
    for col in X_train.columns
    if X_train[col].dtype in ["object", "bool"] and X_train[col].nunique() < 10
]
col_num = [col for col in X_train.columns if X_train[col].dtype in ["float", "int"]]
col_cat, col_num, col_cat + col_num
#%%
transformer_cat = make_pipeline(
    (SimpleImputer()),
    # (OrdinalEncoder(categories=[["Small", "Medium", "Large"], ["First", "Second", "Third"]])),
    (OneHotEncoder()),
)
transformer_num = make_pipeline(
    # (IterativeImputer()),
    (KNNImputer()),
    (PolynomialFeatures()),
    (MinMaxScaler()),
    # (StandardScaler()),
)
preprocessor = make_column_transformer(
    (transformer_cat, col_cat),
    (transformer_num, col_num),
)
preprocessor
#%%
tests = [
    {
        "model": make_pipeline(
            (preprocessor),
            (SelectPercentile()),
            (LinearRegression()),
        ),
        "params": {
            "columntransformer__pipeline-1__simpleimputer__strategy": [
                "constant",
                "most_frequent",
            ],
            # "columntransformer__pipeline-1__onehotencoder__drop": [None, "if_binary"],
            "columntransformer__pipeline-1__onehotencoder__handle_unknown": ["ignore"],
            "columntransformer__pipeline-1__simpleimputer__add_indicator": [
                True,
                False,
            ],
            "columntransformer__pipeline-2__knnimputer__n_neighbors": [1, 3, 5, 7, 9],
            "columntransformer__pipeline-2__knnimputer__add_indicator": [True, False],
            "columntransformer__pipeline-2__polynomialfeatures__degree": range(1, 20),
            "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
            "selectpercentile__score_func": [chi2, f_classif],
        },
    },
]
models = []
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
    models.append(rscv.best_estimator_)
    print("===train============================")
    print(f"{rscv.best_score_ * 100:.2f}%\t{test['model'][2]}\t{rscv.best_params_}")
    print("===params============================")
    display(pd.DataFrame(rscv.cv_results_).sort_values(by="rank_test_score"))
    print("===test============================")
    print(f"test score:{rscv.score(X_test, y_test) * 100:.2f}%")
    print("====end===========================\n")
#%%
final_test = [
    {
        "model": make_pipeline(
            (preprocessor),
            (SelectPercentile()),
            (
                VotingRegressor(
                    estimators=[
                        (
                            list(model.named_steps.keys())[-1],
                            model.named_steps[list(model.named_steps.keys())[-1]],
                        )
                        for model in models
                    ]
                )
            ),
        ),
        "params": {
            "columntransformer__pipeline-1__simpleimputer__strategy": [
                "constant",
                "most_frequent",
            ],
            # "columntransformer__pipeline-1__onehotencoder__drop": [None, "if_binary"],
            "columntransformer__pipeline-1__onehotencoder__handle_unknown": ["ignore"],
            "columntransformer__pipeline-1__simpleimputer__add_indicator": [
                True,
                False,
            ],
            "columntransformer__pipeline-2__knnimputer__n_neighbors": [1, 3, 5, 7, 9],
            "columntransformer__pipeline-2__knnimputer__add_indicator": [True, False],
            "columntransformer__pipeline-2__polynomialfeatures__degree": range(1, 20),
            "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
            "selectpercentile__score_func": [chi2, f_classif],
            # "votingregressor__voting": ["soft", "hard"],
            "votingregressor__weights": [
                (1, 1, 1, 1),
                (1, 1, 1, 0.5),
                (1, 1, 0.5, 1),
                (1, 0.5, 1, 1),
                (0.5, 1, 1, 1),
                (1, 1, 0, 1),
                (1, 0, 1, 1),
                (0, 1, 1, 1),
                (1, 1, 1, 0),
                (1, 1, 0, 0),
                (1, 0, 1, 0),
                (1, 0, 0, 1),
                (0, 1, 1, 0),
                (0, 1, 0, 1),
                (0, 0, 1, 1),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
                (0, 0, 0, 0),
            ],
        },
    }
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
rscv = RandomizedSearchCV(
    estimator=final_test[0]["model"],
    param_distributions=final_test[0]["params"],
    n_jobs=-1,
    cv=cv,
    # scoring="r2",
    scoring="neg_mean_squared_error",
    n_iter=10,
)
rscv.fit(X_train, y_train)
print("===train============================")
print(
    f"{rscv.best_score_ * 100:.2f}%\t{final_test[0]['model'][2]}\t{rscv.best_params_}"
)
print("===params============================")
display(pd.DataFrame(rscv.cv_results_).sort_values(by="rank_test_score"))
print("===test============================")
print(f"test score:{rscv.score(X_test, y_test) * 100:.2f}%")
print("====end===========================\n")
#%%
rscv.best_estimator_
#%%
plt.plot(X, rscv.predict(X), color="red", lineWidth=3)
plt.scatter(X, y)
plt.title(
    "Polynomial Linear Regression using scikit-learn and python 3",
    fontsize=10,
)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%
X_test.groupby(df.columns[0])[df.columns[0]].mean().plot(kind="bar")
#%%
print(r2_score(y_test, rscv.predict(X_test)))
#%%
y_test[:10]
#%%
rscv.predict(X_test)[:10].reshape(-1, 1)
#%%
plot_tree(models[2].named_steps["decisiontreeregressor"], filled=True)
#%%
rscv.best_estimator_.get_params()
#%%
rscv.best_estimator_[:1].fit_transform(X_train, y_train).shape
#%%
rscv.best_estimator_[:2].fit_transform(X_train, y_train).shape
#%%
rscv.best_estimator_[1].get_support()
