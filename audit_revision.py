"""Additional checks for the revised manuscript (public WDBC data only).

Same conventions as audit_experiments.py: same models, same seeds for the
cross-validation repeats (0..9), the ablation repeats (0..4) and the 80/20
splits (0..999). Writes results_revision.json.
"""
import json, warnings
import numpy as np, pandas as pd
from scipy.stats import binom
from scipy.optimize import brentq
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.base import clone
import sklearn

warnings.filterwarnings("ignore")
R = {"sklearn_version": sklearn.__version__}

d = load_breast_cancer(as_frame=True)
X30 = d.data.copy(); y = (d.target == 0).astype(int).values          # 1 = malignant
F5 = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
short = ["radius", "texture", "perimeter", "area", "smoothness"]
X5 = X30[F5]

def models():
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
        "RF": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=2),
        "HGB": HistGradientBoostingClassifier(random_state=0),
    }

# ---- 1. leave-one-feature-out retraining (same protocol as the ablation table) ----
def cv_auc(name, cols):
    a = []
    for r in range(5):
        p = cross_val_predict(clone(models()[name]), X5.iloc[:, cols], y,
                              cv=StratifiedKFold(10, shuffle=True, random_state=r), method="predict_proba")[:, 1]
        a.append(roc_auc_score(y, p))
    return float(np.mean(a))
R["drop_one"] = {}
for name in ["LR", "HGB"]:
    R["drop_one"][name] = {"all five": cv_auc(name, [0, 1, 2, 3, 4])}
    for i, s in enumerate(short):
        R["drop_one"][name]["without " + s] = cv_auc(name, [j for j in range(5) if j != i])
print("drop-one", R["drop_one"])

# ---- 2. single splits: other seed blocks, other split ratios, best-of-k, binomial tail ----
def splits(X, test_size, seeds):
    mdl = models()["LR"]; acc = []; nte = None
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=s)
        acc.append(accuracy_score(yte, clone(mdl).fit(Xtr, ytr).predict(Xte))); nte = len(yte)
    return np.array(acc), nte
base, n_te = splits(X5, 0.2, range(1000))
R["single_split_5feat"] = {"n_test": n_te, "frac_ge_97_seeds_0_999": float((base >= 0.97).mean())}
blocks = []
for b in range(1, 5):
    a, _ = splits(X5, 0.2, range(1000 * b, 1000 * b + 1000)); blocks.append(float((a >= 0.97).mean()))
R["single_split_5feat"]["frac_ge_97_other_seed_blocks"] = blocks
for ts in (0.25, 0.30):
    a, nte = splits(X5, ts, range(1000))
    R["single_split_5feat"][f"test_size_{ts}"] = {"n_test": nte, "mean": float(a.mean()), "frac_ge_97": float((a >= 0.97).mean())}
# expected best of k draws, exact from the empirical distribution of the 1,000 splits
srt = np.sort(base); n = len(srt); cdf = np.arange(1, n + 1) / n
def best_of(k):
    w = cdf ** k - np.concatenate([[0], cdf[:-1]]) ** k
    return float((srt * w).sum())
R["single_split_5feat"]["expected_best_of_k"] = {str(k): best_of(k) for k in (1, 2, 5, 10, 20, 50)}
p_cv = 530 / 569
R["binomial_tail_111_of_114"] = float(1 - binom.cdf(110, 114, p_cv))
print("single split", R["single_split_5feat"], R["binomial_tail_111_of_114"])

# ---- 3. calibration: joint intercept, calibration-in-the-large, clipping sensitivity ----
def oof(X, name):
    P = [cross_val_predict(clone(models()[name]), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=r),
                           method="predict_proba")[:, 1] for r in range(10)]
    return np.mean(P, 0)
def recal(p, eps):
    q = np.clip(p, eps, 1 - eps); z = np.log(q / (1 - q))
    lr = LogisticRegression(C=1e9, max_iter=10000).fit(z.reshape(-1, 1), y)
    slope, inter = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    # calibration-in-the-large: intercept with the slope fixed at 1 (logit offset)
    f = lambda a: (1 / (1 + np.exp(-(a + z)))).sum() - y.sum()
    citl = float(brentq(f, -20, 20))
    return slope, inter, citl
R["calibration"] = {}
for fs, X in (("5feat", X5), ("30feat", X30)):
    R["calibration"][fs] = {}
    for name in models():
        p = oof(X, name)
        out = {"mean_predicted": float(p.mean()), "prevalence": float(y.mean()),
               "n_prob_exactly_0_or_1": int(((p == 0) | (p == 1)).sum())}
        for eps in (1e-6, 1e-3, 1e-2):
            s, i, c = recal(p, eps); out[f"clip_{eps:g}"] = {"slope": s, "intercept_joint": i, "intercept_slope_fixed_1": c}
        R["calibration"][fs][name] = out
        print(fs, name, out)

# ---- 4. fused-table control: what the classifier needs in order to find the source ----
rng = np.random.default_rng(20261116)
K = 39
cols = {"breast": list(range(0, 5)), "fourth": [5, 6, 7, 37, 38], "lung": list(range(8, 28)) + [37, 38], "prostate": list(range(28, 37))}
sizes = {"breast": 569, "prostate": 100, "lung": 4000, "fourth": 4000}      # two large blocks subsampled
Xs, ys = [], []
for li, k in enumerate(sizes):
    M = np.full((sizes[k], K), np.nan); M[:, cols[k]] = rng.normal(size=(sizes[k], len(cols[k])))
    Xs.append(M); ys += [li] * sizes[k]
Xf = np.vstack(Xs); yf = np.array(ys); Xz = np.nan_to_num(Xf, nan=0.0)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
acc = lambda m, X: float((cross_val_predict(m, X, yf, cv=skf) == yf).mean())
pattern = (~np.isnan(Xf)).astype(int)
rule = np.array([{tuple(pattern[yf == c][0]): c for c in range(4)}[tuple(r)] for r in pattern])
R["fusion_control"] = {
    "sizes": sizes,
    "majority_share_subsampled": float(max(np.bincount(yf)) / len(yf)),
    "lookup_rule_on_missingness_pattern": float((rule == yf).mean()),
    "HGB_native_missing": acc(HistGradientBoostingClassifier(random_state=0), Xf),
    "LR_zero_filled_with_indicators": acc(LogisticRegression(max_iter=2000), np.hstack([Xz, np.isnan(Xf)])),
    "RF_zero_filled_no_indicators": acc(RandomForestClassifier(200, random_state=0, n_jobs=2), Xz),
    "LR_zero_filled_no_indicators": acc(LogisticRegression(max_iter=2000), Xz),
}
print("fusion control", R["fusion_control"])

json.dump(R, open("results_revision.json", "w"), indent=1)
print("done")
