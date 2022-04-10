#%%
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from numpy.core.fromnumeric import size
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
df, y_label = load_iris(as_frame=True)["frame"], "target"
df, y_label = pd.read_csv("../.data/kaggle.csv"), "output"
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
sns.pairplot(
    data=df,
    hue=y_label,
    vars=[df.columns[i] for i in range(df.shape[1]) if i in [2, 7, 10, 13]],
    kind="reg",
    diag_kind="kde",
    size=5,
    palette="husl",
)
#%%
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
ax = ax.flatten()
for i, (col, value) in enumerate(df.items()):
    # sns.catplot(x=y_label, y=col, data=df, ax=ax[i], kind="swarm")
    sns.boxplot(y=col, data=df, ax=ax[i])
plt.tight_layout()
#%%
sns.boxplot(x=y_label, y=df.columns[0], data=df)
#%%
sns.catplot(x=y_label, y=df.columns[0], data=df, kind="swarm")
#%%
sns.displot(df[y_label], kind="kde")
#%%
sns.FacetGrid(df, row=df.columns[3], col=df.columns[1], hue=y_label).map(
    plt.hist, df.columns[2], alpha=0.75
).add_legend()
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
    stratify=y,
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
            (LogisticRegression()),
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
            "columntransformer__pipeline-2__polynomialfeatures__degree": range(1, 2),
            "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
            "selectpercentile__score_func": [chi2, f_classif],
            "logisticregression__penalty": ["l1", "l2"],
            "logisticregression__C": range(1, 10),
            "logisticregression__solver": ["liblinear"],
        },
    },
    {
        "model": make_pipeline(
            (preprocessor),
            (SelectPercentile()),
            (GaussianNB()),
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
            "columntransformer__pipeline-2__polynomialfeatures__degree": range(1, 2),
            "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
            "selectpercentile__score_func": [chi2, f_classif],
        },
    },
    {
        "model": make_pipeline(
            (preprocessor),
            (SelectPercentile()),
            (DecisionTreeClassifier()),
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
            "columntransformer__pipeline-2__polynomialfeatures__degree": range(1, 2),
            "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
            "selectpercentile__score_func": [chi2, f_classif],
            "decisiontreeclassifier__criterion": ["gini", "entropy"],
            "decisiontreeclassifier__splitter": ["best", "random"],
            "decisiontreeclassifier__max_depth": [5, 10, 20, 50, 100, 200],
            "decisiontreeclassifier__min_samples_split": [2, 5, 10, 20, 50, 100, 200],
            "decisiontreeclassifier__min_samples_leaf": [5, 10, 20, 50, 100, 200],
            "decisiontreeclassifier__max_features": ["auto", "sqrt", "log2"],
            "decisiontreeclassifier__ccp_alpha": [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ],
        },
    },
    {
        "model": make_pipeline(
            (preprocessor),
            (SelectPercentile()),
            (RandomForestClassifier()),
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
            "columntransformer__pipeline-2__polynomialfeatures__degree": range(1, 2),
            "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
            "selectpercentile__score_func": [chi2, f_classif],
            "randomforestclassifier__n_estimators": [100, 150, 200, 500],
            "randomforestclassifier__criterion": ["gini", "entropy"],
            "randomforestclassifier__max_depth": [5, 10, 20, 50, 100, 200],
            "randomforestclassifier__min_samples_split": [2, 5, 10, 20, 50, 100, 200],
            "randomforestclassifier__min_samples_leaf": [5, 10, 20, 50, 100, 200],
            "randomforestclassifier__max_features": ["auto", "sqrt", "log2"],
        },
    },
]
models = []
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
                VotingClassifier(
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
            "columntransformer__pipeline-2__polynomialfeatures__degree": range(1, 2),
            "selectpercentile__percentile": [i * 10 for i in range(1, 10)],
            "selectpercentile__score_func": [chi2, f_classif],
            "votingclassifier__voting": ["soft"],
            # "votingclassifier__voting": ["soft", "hard"],
            "votingclassifier__weights": [
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
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
rscv = RandomizedSearchCV(
    estimator=final_test[0]["model"],
    param_distributions=final_test[0]["params"],
    n_jobs=-1,
    cv=cv,
    scoring="accuracy",
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
plot_confusion_matrix(rscv, X_test, y_test, values_format="d")
#%%
plot_roc_curve(rscv, X_test, y_test)
#%%
print(roc_auc_score(y_test, rscv.predict(X_test)))
#%%
print(classification_report(y_test, rscv.predict(X_test)))
#%%
y_test[:10]
#%%
rscv.predict(X_test)[:10].reshape(-1, 1)
#%%
rscv.predict_proba(X_test)[:10, :]
#%%
plot_tree(models[2].named_steps["decisiontreeclassifier"], filled=True)
#%%
rscv.best_estimator_.get_params()
#%%
rscv.best_estimator_[:1].fit_transform(X_train, y_train).shape
#%%
rscv.best_estimator_[:2].fit_transform(X_train, y_train).shape
#%%
rscv.best_estimator_[1].get_support()
