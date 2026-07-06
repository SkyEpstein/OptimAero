"""Generate the key figures for the methods-and-results writeup from the saved result JSONs."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_REPO, "docs", "figures")
os.makedirs(FIG, exist_ok=True)
bk = json.load(open(os.path.join(_REPO, "results", "phase2_bakeoff.json")))

# --- Fig 1: predictor bake-off R² (new-geometry) ---
preds = {k: v for k, v in bk["predictors"].items()}
order = sorted(preds, key=lambda k: preds[k]["r2"][0], reverse=True)
x = np.arange(len(order)); w = 0.26
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (lab, c) in enumerate(zip(["Cl", "Cd", "Cm"], ["#2563eb", "#dc2626", "#16a34a"])):
    ax.bar(x + (i - 1) * w, [preds[k]["r2"][i] for k in order], w, label=lab, color=c)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("R²  (new-geometry, held-out families)"); ax.set_ylim(0.4, 1.0)
ax.set_title("Predictor bake-off — honest generalization (MLP wins)")
ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_predictor_bakeoff.png"), dpi=150); plt.close(fig)

# --- Fig 2: selective-prediction curve (winner) ---
conf = bk["confidence"][bk["winner"]]
cov = [100, 50, 25, 10]
fig, ax = plt.subplots(figsize=(7, 4.2))
for lab, c in zip(["Cl", "Cd", "Cm"], ["#2563eb", "#dc2626", "#16a34a"]):
    rmse = [conf[lab]["coverage"][str(q)]["rmse"] for q in cov]
    ax.plot(cov, np.array(rmse) / rmse[0], "o-", label=lab, color=c)
ax.set_xlabel("coverage (%) — keeping the most-confident predictions")
ax.set_ylabel("RMSE (normalized to 100% coverage)")
ax.set_title("Confidence model works: error falls as we keep only trusted predictions")
ax.invert_xaxis(); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_selective_prediction.png"), dpi=150); plt.close(fig)

# --- Fig 3: the verification finding ---
fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ["baseline\n(real)", "surrogate\nCLAIM", "XFOIL-verified\noptimum"]
vals = [100, 441, 129]; cols = ["#94a3b8", "#f59e0b", "#16a34a"]
b = ax.bar(bars, vals, color=cols)
ax.bar_label(b, fmt="L/D=%.0f", padding=3)
ax.set_ylabel("L/D"); ax.set_title("Trust before speed: verification catches the 3.4× over-promise")
ax.text(1, 300, "optimizer exploited\na surrogate blind spot", ha="center", color="#b45309", fontsize=9)
ax.text(2, 175, "+29% over baseline,\nXFOIL-confirmed", ha="center", color="#166534", fontsize=9)
ax.grid(axis="y", alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_verification.png"), dpi=150); plt.close(fig)

print("wrote:", *sorted(os.listdir(FIG)))
