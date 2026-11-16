"""Supplementary checks added after independent review. Run after exp.py: python3 exp2.py"""
import json, warnings, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(7)
R = json.load(open("results.json"))
d = load_breast_cancer(as_frame=True); y = (d.target == 0).astype(int).values
X5 = d.data[["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]]

def slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); z = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(z, y).coef_[0, 0])

M = {"LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
     "LR-unpenalised": make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000)),
     "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
     "RF": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=2),
     "HGB": HistGradientBoostingClassifier(random_state=0)}
S, AVG = {}, {}
for name, mdl in M.items():
    P = np.array([cross_val_predict(clone(mdl), X5, y, cv=StratifiedKFold(10, shuffle=True, random_state=r), method="predict_proba")[:, 1] for r in range(10)])
    acc = [accuracy_score(y, (p >= .5).astype(int)) for p in P]; sl = [slope(y, p) for p in P]
    AVG[name] = P.mean(0)
    S[name] = dict(acc_mean=float(np.mean(acc)), acc_sd=float(np.std(acc)), acc_min=float(min(acc)), acc_max=float(max(acc)),
                   slope_mean=float(np.mean(sl)), slope_min=float(min(sl)), slope_max=float(max(sl)))
    print(name, S[name])
R["per_repeat_5feat"] = S
# paired bootstrap of the accuracy difference LR - RF on the same patients
a = (AVG["LR"] >= .5).astype(int) == y; b = (AVG["RF"] >= .5).astype(int) == y
diff = [a[ix].mean() - b[ix].mean() for ix in (RNG.integers(0, len(y), len(y)) for _ in range(5000))]
R["paired_LR_minus_RF_acc"] = dict(point=float(a.mean() - b.mean()), lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)))
print(R["paired_LR_minus_RF_acc"])

# E5 corrected: Age and Tumor_Size_mm are both shared by the lung and fourth sources (191,155 observations each)
excl = {"breast": 5, "fourth": 3, "lung": 20, "prostate": 9}; n = {"breast": 569, "prostate": 100, "lung": 4000, "fourth": 4000}
offs, c = {}, 0
for k in ["breast", "fourth", "lung", "prostate"]:
    offs[k] = (c, c + excl[k]); c += excl[k]
shared = [37, 38]
rows, lab = [], []
for li, k in enumerate(["breast", "prostate", "lung", "fourth"]):
    A = np.full((n[k], 39), np.nan); s, e = offs[k]; A[:, s:e] = RNG.normal(size=(n[k], e - s))
    if k in ("lung", "fourth"): A[:, shared] = RNG.normal(size=(n[k], 2))
    rows.append(A); lab += [li] * n[k]
Xf, yf = np.vstack(rows), np.array(lab); Xz = np.nan_to_num(Xf, nan=0.0); skf = StratifiedKFold(5, shuffle=True, random_state=0)
R["fusion"]["noise_control_acc"] = dict(
    HGB_native_missing=float((cross_val_predict(HistGradientBoostingClassifier(random_state=0), Xf, yf, cv=skf) == yf).mean()),
    LR_with_indicators=float((cross_val_predict(LogisticRegression(max_iter=2000), np.hstack([Xz, np.isnan(Xf)]), yf, cv=skf) == yf).mean()),
    RF_zero_filled_no_indicators=float((cross_val_predict(RandomForestClassifier(200, random_state=0, n_jobs=2), Xz, yf, cv=skf) == yf).mean()))
R["fusion"]["structure"] = dict(exclusive=excl, shared_lung_fourth=["Age", "Tumor_Size_mm"])
print(R["fusion"]["noise_control_acc"])
json.dump(R, open("results.json", "w"), indent=1)

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(7.0, 2.2))
order = [("breast", 569, 0, 5), ("fourth source", 167497, 5, 8), ("lung", 23658, 8, 28), ("prostate", 100, 28, 37)]
B = np.zeros((4, 39))
for i, (_, _, s, e) in enumerate(order): B[i, s:e] = 1
B[1, 37:] = 1; B[2, 37:] = 1
ax.imshow(B, cmap=matplotlib.colors.ListedColormap(["#f0f0f0", "#1f4e79"]), aspect="auto")
ax.set_yticks(range(4)); ax.set_yticklabels([f"{k} (n = {v:,})" for k, v, _, _ in order])
ax.set_xticks([2, 6, 17.5, 32, 37.5]); ax.set_xticklabels(["5 cols", "3 cols", "20 cols", "9 cols", "Age,\ntumour size"])
ax.set_xlabel("Columns of the fused table (dark: observed; light: structurally missing)")
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks(np.arange(-.5, 39, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True); ax.grid(which="minor", color="white", lw=.6); ax.tick_params(which="minor", length=0)
plt.tight_layout(); plt.savefig("fig_blocks.pdf")
