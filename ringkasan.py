"""Ringkasan statistik dari cv_results.csv dan holdout_results.csv -> ringkasan.json"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
cv = pd.read_csv("cv_results.csv"); ho = pd.read_csv("holdout_results.csv")
M5 = ["Regresi logistik", "SVM (RBF)", "k-NN (k = 5)", "Random Forest", "Gradient Boosting"]
out = {}
# RQ1
t1 = cv.groupby("model")[["akurasi","macro_f1","auc"]].agg(["mean","std"])
out["cv"] = {m: {k: [float(t1.loc[m,(k,"mean")]), float(t1.loc[m,(k,"std")])] for k in ["akurasi","macro_f1","auc"]} for m in t1.index}
out["cv_n"] = int(cv.groupby("model").size().iloc[0])
piv = cv.pivot_table(index=["ulangan","lipatan"], columns="model", values="akurasi")
def paired(a,b):
    d = piv[a]-piv[b]
    return {"mean_diff": float(d.mean()), "a_lebih": int((d>0).sum()), "sama": int((d==0).sum()), "b_lebih": int((d<0).sum()), "n": int(len(d))}
out["paired"] = {"LR-SVM": paired("Regresi logistik","SVM (RBF)"), "LR-RF": paired("Regresi logistik","Random Forest"),
                 "LR-GB": paired("Regresi logistik","Gradient Boosting"), "LR-kNN": paired("Regresi logistik","k-NN (k = 5)")}
# RQ2
runmean = cv.groupby(["model","ulangan"])["akurasi"].mean().reset_index()
h = ho.groupby("model")["akurasi"].agg(["median","mean","std","min","max"])
out["holdout"] = {m: {"median": float(h.loc[m,"median"]), "mean": float(h.loc[m,"mean"]), "sd": float(h.loc[m,"std"]), "min": float(h.loc[m,"min"]), "max": float(h.loc[m,"max"])} for m in h.index}
out["cvrun"] = {m: {"sd": float(g["akurasi"].std()), "min": float(g["akurasi"].min()), "max": float(g["akurasi"].max())} for m, g in runmean.groupby("model")}
# RQ3
hp = ho.pivot_table(index="split", columns="model", values="akurasi")[M5]
mx = hp.max(axis=1)
cobest = {m: int((hp[m]==mx).sum()) for m in M5}
strict = {}
for m in M5:
    others = hp.drop(columns=m).max(axis=1)
    strict[m] = int((hp[m]>others).sum())
cvbest = max(M5, key=lambda m: out["cv"][m]["akurasi"][0])
cv_rank = pd.Series({m: out["cv"][m]["akurasi"][0] for m in M5}).rank(ascending=False)
rho = [spearmanr(cv_rank.values, hp.loc[s, M5].rank(ascending=False).values)[0] for s in hp.index]
tops = {}
for s in hp.index:
    best = tuple(sorted([m for m in M5 if hp.loc[s,m]==mx[s]]))
    tops[best] = tops.get(best,0)+1
out["rq3"] = {"cvbest": cvbest, "cobest": cobest, "strict": strict, "rho_mean": float(np.nanmean(rho)), "rho_min": float(np.nanmin(rho)), "rho_max": float(np.nanmax(rho)),
    "lr_vs_svm": {"lr_lebih": int((hp["Regresi logistik"]>hp["SVM (RBF)"]).sum()), "sama": int((hp["Regresi logistik"]==hp["SVM (RBF)"]).sum()), "svm_lebih": int((hp["Regresi logistik"]<hp["SVM (RBF)"]).sum())},
    "n_splits": int(len(hp)), "n_distinct_top_sets": len(tops), "top_sets": {" + ".join(k): v for k, v in sorted(tops.items(), key=lambda kv: -kv[1])}}
# Ringkasan lain
out["baseline_acc"] = float(cv[cv.model.str.startswith("Baseline")]["akurasi"].mean())
json.dump(out, open("ringkasan.json","w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
