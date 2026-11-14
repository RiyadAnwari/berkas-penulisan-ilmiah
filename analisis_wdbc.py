"""Analisis JS10: evaluasi enam model pada Breast Cancer Wisconsin (Diagnostic).
Dataset: salinan UCI WDBC yang disertakan scikit-learn (sklearn.datasets.load_breast_cancer).
Keluaran: cv_results.csv, holdout_results.csv, ringkasan.json, log_analisis.txt
"""
import json, sys, numpy as np, pandas as pd, sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 42
X, y = load_breast_cancer(return_X_y=True)
print("scikit-learn", sklearn.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)
print("n =", X.shape[0], "| fitur =", X.shape[1], "| kelas 0 (maligna) =", int((y == 0).sum()), "| kelas 1 (jinak) =", int((y == 1).sum()))

def models():
    return {
        "Baseline (kelas mayoritas)": DummyClassifier(strategy="most_frequent"),
        "Regresi logistik": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "k-NN (k = 5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }

def score(m, Xtr, ytr, Xte, yte):
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    if hasattr(m, "predict_proba"):
        s = m.predict_proba(Xte)[:, 1]
    else:
        s = m.decision_function(Xte)
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro"), roc_auc_score(yte, s)

# Protokol A: validasi silang berstrata 5 lipatan x 10 ulangan
rows = []
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
for i, (tr, te) in enumerate(rskf.split(X, y)):
    rep, fold = i // 5 + 1, i % 5 + 1
    for name, m in models().items():
        a, f, u = score(m, X[tr], y[tr], X[te], y[te])
        rows.append((name, rep, fold, len(te), a, f, u))
cv = pd.DataFrame(rows, columns=["model", "ulangan", "lipatan", "n_uji", "akurasi", "macro_f1", "auc"])
cv.to_csv("cv_results.csv", index=False)

# Protokol B: 100 holdout 80/20 berstrata
rows = []
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=SEED)
for i, (tr, te) in enumerate(sss.split(X, y)):
    for name, m in models().items():
        a, f, u = score(m, X[tr], y[tr], X[te], y[te])
        rows.append((name, i + 1, len(te), a, f, u))
ho = pd.DataFrame(rows, columns=["model", "split", "n_uji", "akurasi", "macro_f1", "auc"])
ho.to_csv("holdout_results.csv", index=False)
print("n uji CV (min-maks):", cv.n_uji.min(), cv.n_uji.max(), "| n uji holdout:", ho.n_uji.unique())
print(cv.groupby("model")[["akurasi", "macro_f1", "auc"]].agg(["mean", "std"]).round(4).to_string())
print(ho.groupby("model")["akurasi"].agg(["median", "mean", "std", "min", "max"]).round(4).to_string())
