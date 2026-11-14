"""Gambar 1: sebaran akurasi 100 holdout 80/20 per model + rerata CV 5-lipatan x 10 (data: holdout_results.csv, cv_results.csv)"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "DejaVu Sans"; rcParams["font.size"] = 10
ho = pd.read_csv("holdout_results.csv"); cv = pd.read_csv("cv_results.csv")
M = ["Regresi logistik", "SVM (RBF)", "k-NN (k = 5)", "Random Forest", "Gradient Boosting"]
lab = ["Regresi\nlogistik", "SVM\nRBF", "k-NN\n(k = 5)", "Random\nForest", "Gradient\nBoosting"]
cvmean = [cv[cv.model == m]["akurasi"].mean() * 100 for m in M]
fig, ax = plt.subplots(figsize=(5.4, 3.3), dpi=300)
data = [ho[ho.model == m]["akurasi"].values * 100 for m in M]
bp = ax.boxplot(data, positions=range(len(M)), widths=0.5, showfliers=False, whis=(0, 100), patch_artist=True,
                medianprops=dict(color="black", lw=1.6), boxprops=dict(facecolor="#D9E2EC", edgecolor="#334E68"), whiskerprops=dict(color="#334E68"), capprops=dict(color="#334E68"))
rng = np.random.default_rng(7)
for i, d in enumerate(data):
    ax.scatter(i + rng.uniform(-0.17, 0.17, len(d)), d, s=9, color="#486581", alpha=0.45, zorder=3, linewidths=0)
ax.scatter(range(len(M)), cvmean, marker="D", s=44, color="#D64545", edgecolor="black", linewidths=0.6, zorder=5, label="Rerata validasi silang (50 lipatan)")
for i, v in enumerate(cvmean):
    ax.text(i + 0.29, v, f"{v:.1f}".replace(".", ","), fontsize=8.5, color="#9B1C1C", fontweight="bold", va="center", ha="left", zorder=6)
ax.set_xlim(-0.6, 4.85); ax.set_xticks(range(len(M))); ax.set_xticklabels(lab)
ax.set_ylim(89.5, 100.8); ax.set_yticks(range(90, 101, 2))
ax.set_ylabel("Akurasi pada data uji (%)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.0f}"))
ax.grid(axis="y", color="#CBD2D9", lw=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
from matplotlib.lines import Line2D
h = [Line2D([0],[0], marker="o", color="none", markerfacecolor="#486581", alpha=0.6, markersize=5, label="Satu titik = satu holdout 80/20 (n = 100)"),
     Line2D([0],[0], color="black", lw=1.6, label="Median; kotak = kuartil; kumis = nilai terkecil dan terbesar"),
     Line2D([0],[0], marker="D", color="none", markerfacecolor="#D64545", markeredgecolor="black", markersize=6, label="Rerata validasi silang 5 lipatan × 10 ulangan")]
ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.50), frameon=False, fontsize=8, ncol=1)
fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.36)
fig.savefig("gambar1_sebaran_holdout.png", dpi=300)
print("ok", min(min(d) for d in data), max(max(d) for d in data))
