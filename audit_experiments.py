"""Reproducible experiments for the audit manuscript. Run: python3 exp.py  (scikit-learn only)."""
import json, warnings, itertools
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20261116)
R = {}

d = load_breast_cancer(as_frame=True)
X30 = d.data.copy(); y = (d.target == 0).astype(int).values      # 1 = malignant
F5 = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
X5 = X30[F5]
R["data"] = dict(n=int(len(y)), malignant=int(y.sum()), benign=int((1 - y).sum()))

def models():
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
        "RF": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=2),
        "HGB": HistGradientBoostingClassifier(random_state=0),
    }

def cal(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])

def ece(y, p, bins=10):
    e = 0; edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        k = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if k.sum(): e += k.mean() * abs(y[k].mean() - p[k].mean())
    return float(e)

def metrics(y, p):
    yh = (p >= 0.5).astype(int)
    tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
    s, i = cal(y, p)
    return dict(acc=accuracy_score(y, yh), auc=roc_auc_score(y, p), sens=tp / (y == 1).sum(),
                spec=tn / (y == 0).sum(), brier=brier_score_loss(y, p), slope=s, intercept=i, ece=ece(y, p))

def boot_ci(y, p, B=2000):
    out = {k: [] for k in ["acc", "auc", "sens", "spec", "brier", "slope", "ece"]}
    n = len(y)
    for _ in range(B):
        ix = RNG.integers(0, n, n)
        if y[ix].min() == y[ix].max(): continue
        m = metrics(y[ix], p[ix])
        for k in out: out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}

# ---------- E1: repeated stratified 10-fold CV, fixed default hyper-parameters ----------
OOF = {}
R["cv"] = {}
for fs_name, X in [("5feat", X5), ("30feat", X30)]:
    R["cv"][fs_name] = {}
    for name, mdl in models().items():
        P = []
        for r in range(10):
            skf = StratifiedKFold(10, shuffle=True, random_state=r)
            P.append(cross_val_predict(clone(mdl), X, y, cv=skf, method="predict_proba")[:, 1])
        P = np.array(P); pbar = P.mean(0)
        OOF[(fs_name, name)] = pbar
        per = [metrics(y, p) for p in P]
        m = metrics(y, pbar)
        R["cv"][fs_name][name] = dict(point={k: float(v) for k, v in m.items()}, ci=boot_ci(y, pbar),
                                      sd_repeats={k: float(np.std([q[k] for q in per])) for k in m})
        print(fs_name, name, {k: round(float(v), 3) for k, v in m.items()})
R["majority_acc"] = float(max(y.mean(), 1 - y.mean()))

# ---------- E2: how much can a single 80/20 split move the headline accuracy? ----------
def single_splits(X, mdl, n):
    acc = []
    for s in range(n):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
        acc.append(accuracy_score(yte, clone(mdl).fit(Xtr, ytr).predict(Xte)))
    return np.array(acc)
ss = {}
for fs_name, X in [("5feat", X5), ("30feat", X30)]:
    a = single_splits(X, models()["LR"], 1000); ss[fs_name] = a
    best20 = [a[RNG.integers(0, len(a), 20)].max() for _ in range(5000)]
    R.setdefault("single_split", {})[fs_name] = dict(
        mean=float(a.mean()), sd=float(a.std()), min=float(a.min()), p2_5=float(np.percentile(a, 2.5)),
        median=float(np.median(a)), p97_5=float(np.percentile(a, 97.5)), max=float(a.max()),
        frac_ge_97=float((a >= 0.97).mean()), frac_ge_95=float((a >= 0.95).mean()),
        expected_best_of_20=float(np.mean(best20)), n_test=114)
print("single split", R["single_split"])

# ---------- E3: optimistic protocol (tuning on the test set) vs nested protocol ----------
grid = {"svc__C": [0.1, 1, 10, 100, 1000], "svc__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1]}
res = {"5feat": {"honest": [], "optimistic": []}, "30feat": {"honest": [], "optimistic": []}}
for fs_name, X in [("5feat", X5), ("30feat", X30)]:
    for s in range(100):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1000 + s)
        base = make_pipeline(StandardScaler(), SVC())
        gs = GridSearchCV(base, grid, cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=2).fit(Xtr, ytr)
        res[fs_name]["honest"].append(gs.score(Xte, yte))
        best = 0
        for C, g in itertools.product(grid["svc__C"], grid["svc__gamma"]):
            best = max(best, clone(base).set_params(svc__C=C, svc__gamma=g).fit(Xtr, ytr).score(Xte, yte))
        res[fs_name]["optimistic"].append(best)
    h, o = np.array(res[fs_name]["honest"]), np.array(res[fs_name]["optimistic"])
    R.setdefault("tuning_leak", {})[fs_name] = dict(honest_mean=float(h.mean()), optimistic_mean=float(o.mean()),
        gap_mean=float((o - h).mean()), gap_p97_5=float(np.percentile(o - h, 97.5)), gap_max=float((o - h).max()),
        optimistic_max=float(o.max()))
print("tuning leak", R["tuning_leak"])

# ---------- E4: explanations on the 5-feature model family ----------
short = ["radius", "texture", "perimeter", "area", "smoothness"]
R["corr"] = X5.corr().round(3).values.tolist()
# (a) standardized LR coefficients with bootstrap
Z = StandardScaler().fit_transform(X5)
coef = LogisticRegression(max_iter=5000).fit(Z, y).coef_[0]
bc = []
for _ in range(1000):
    ix = RNG.integers(0, len(y), len(y))
    bc.append(LogisticRegression(max_iter=5000).fit(StandardScaler().fit_transform(X5.values[ix]), y[ix]).coef_[0])
bc = np.array(bc)
R["lr_coef"] = {s: dict(coef=float(c), lo=float(np.percentile(bc[:, i], 2.5)), hi=float(np.percentile(bc[:, i], 97.5)),
                        frac_sign_flip=float((np.sign(bc[:, i]) != np.sign(c)).mean())) for i, (s, c) in enumerate(zip(short, coef))}
R["lr_top1_freq"] = {short[i]: float((np.abs(bc).argmax(1) == i).mean()) for i in range(5)}

# (b) held-out permutation importance (AUROC drop), single-feature and grouped
groups = {"size (radius, perimeter, area)": [0, 2, 3], "texture": [1], "smoothness": [4]}
def perm_imp(mdl, cols_sets, n_rep=5, n_perm=5):
    runs = []
    for r in range(n_rep):
        skf = StratifiedKFold(10, shuffle=True, random_state=100 + r); imp = np.zeros(len(cols_sets)); cnt = 0
        for tr, te in skf.split(X5, y):
            m = clone(mdl).fit(X5.iloc[tr], y[tr]); Xte = X5.iloc[te].values; base = roc_auc_score(y[te], m.predict_proba(X5.iloc[te])[:, 1])
            for j, cols in enumerate(cols_sets):
                dsum = 0
                for _ in range(n_perm):
                    Xp = Xte.copy(); perm = RNG.permutation(len(te)); Xp[:, cols] = Xp[perm][:, cols]
                    dsum += base - roc_auc_score(y[te], m.predict_proba(pd.DataFrame(Xp, columns=F5))[:, 1])
                imp[j] += dsum / n_perm
            cnt += 1
        runs.append(imp / cnt)
    return np.array(runs)
PI, PG = {}, {}
for name, mdl in models().items():
    PI[name] = perm_imp(mdl, [[i] for i in range(5)]); PG[name] = perm_imp(mdl, list(groups.values()))
R["perm_single"] = {n: dict(mean=dict(zip(short, v.mean(0).round(4).tolist())), sd=dict(zip(short, v.std(0).round(4).tolist())),
                            top1=short[int(v.mean(0).argmax())],
                            top1_freq_over_repeats={short[i]: float((v.argmax(1) == i).mean()) for i in range(5)}) for n, v in PI.items()}
R["perm_group"] = {n: dict(zip(groups.keys(), v.mean(0).round(4).tolist())) for n, v in PG.items()}
names = list(PI)
R["agreement_single"] = {f"{a}-{b}": float(spearmanr(PI[a].mean(0), PI[b].mean(0))[0]) for a, b in itertools.combinations(names, 2)}
R["agreement_group_top1"] = {n: list(groups.keys())[int(PG[n].mean(0).argmax())] for n in names}
R["top1_single"] = {n: short[int(PI[n].mean(0).argmax())] for n in names}

# (c) ablation by retraining: which feature groups actually carry the signal?
subsets = {"all five": [0, 1, 2, 3, 4], "size only": [0, 2, 3], "radius only": [0], "perimeter only": [2], "area only": [3],
           "texture only": [1], "smoothness only": [4], "radius + texture": [0, 1], "radius + texture + smoothness": [0, 1, 4],
           "texture + smoothness (no size)": [1, 4]}
R["ablation"] = {}
for sname, cols in subsets.items():
    R["ablation"][sname] = {}
    for name in ["LR", "HGB"]:
        aucs = []
        for r in range(5):
            p = cross_val_predict(clone(models()[name]), X5.iloc[:, cols], y, cv=StratifiedKFold(10, shuffle=True, random_state=r), method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(y, p))
        R["ablation"][sname][name] = float(np.mean(aucs))
print("ablation", R["ablation"])

# ---------- E5: fused-table shortcut, noise-feature control ----------
# Block structure recovered from the fitted scaler of the prototype (n_samples_seen_ per column).
blocks = {"breast": dict(n=569, k=5), "prostate": dict(n=100, k=9), "lung": dict(n=23658, k=20), "fourth": dict(n=167497, k=4)}
tot = sum(b["n"] for b in blocks.values())
R["fusion"] = dict(total_rows=tot, majority_share=167497 / tot, shares={k: b["n"] / tot for k, b in blocks.items()})
def fused(nmap):
    rows, lab = [], []; start = 0; K = 39
    offs = {}; c = 0
    for k in ["breast", "fourth", "lung", "prostate"]:
        offs[k] = (c, c + blocks[k]["k"]); c += blocks[k]["k"]
    age = 38
    for li, k in enumerate(blocks):
        M = np.full((nmap[k], K), np.nan); a, b = offs[k]; M[:, a:b] = RNG.normal(size=(nmap[k], b - a))
        if k in ("lung", "fourth"): M[:, age] = RNG.normal(size=nmap[k])
        rows.append(M); lab += [li] * nmap[k]
    return np.vstack(rows), np.array(lab)
Xf, yf = fused({"breast": 569, "prostate": 100, "lung": 4000, "fourth": 4000})
skf = StratifiedKFold(5, shuffle=True, random_state=0)
p_hgb = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Xf, yf, cv=skf)
Xz = np.nan_to_num(Xf, nan=0.0)                       # zero-fill after standardisation, as a fused scaler would produce
p_lr = cross_val_predict(LogisticRegression(max_iter=2000), np.hstack([Xz, np.isnan(Xf)]), yf, cv=skf)
p_lr0 = cross_val_predict(RandomForestClassifier(200, random_state=0, n_jobs=2), Xz, yf, cv=skf)
R["fusion"]["noise_control_acc"] = dict(HGB_native_missing=float((p_hgb == yf).mean()), LR_with_indicators=float((p_lr == yf).mean()),
                                        RF_zero_filled_no_indicators=float((p_lr0 == yf).mean()))
print("fusion", R["fusion"])

# ---------- small-sample arithmetic used in the text ----------
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]
R["wilson"] = {"17/20": wilson(17, 20), "111/114": wilson(111, 114)}
R["breakhis"] = dict(train=1148, test=545, train_benign=371, train_malignant=777, test_benign=176, test_malignant=369,
                     patients_train=81, patients_test=80, overlap=80, test_majority=369 / 545)

json.dump(R, open("results.json", "w"), indent=1)

# ---------- figures ----------
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#1f4e79", "#c0504d", "#7f7f7f", "#4f9d69"
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for a, k, t in zip(ax, ["5feat", "30feat"], ["Five features (prototype subset)", "Thirty features (full WDBC)"]):
    v = ss[k] * 100; a.hist(v, bins=np.arange(v.min() - .44, v.max() + .9, .877), color=C1, alpha=.85)
    cvv = R["cv"][k]["LR"]["point"]["acc"] * 100
    a.axvline(cvv, color=C2, lw=1.5, label=f"repeated 10-fold CV: {cvv:.1f}%"); a.axvline(97, color=C3, ls="--", lw=1.2, label="97% (project notes)")
    a.set_title(t, fontsize=9); a.set_xlabel("Test accuracy of one random 80/20 split (%)"); a.legend(frameon=False, fontsize=7, loc="upper left")
ax[0].set_ylabel("Number of splits (of 1,000)")
plt.tight_layout(); plt.savefig("fig_splits.pdf"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for a, k, t in zip(ax, ["5feat", "30feat"], ["Five features", "Thirty features"]):
    a.plot([0, 1], [0, 1], color=C3, lw=.8, ls=":")
    for (name, col, mk) in [("LR", C1, "o"), ("SVM", C4, "s"), ("RF", C2, "^"), ("HGB", "#8064a2", "D")]:
        p = OOF[(k, name)]; q = pd.qcut(p, 8, duplicates="drop"); g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
        a.plot(g.p, g.y, marker=mk, ms=3.5, lw=1, color=col, label=f"{name} (slope {R['cv'][k][name]['point']['slope']:.2f})")
    a.set_title(t, fontsize=9); a.set_xlabel("Predicted probability of malignancy"); a.legend(frameon=False, fontsize=7)
ax[0].set_ylabel("Observed proportion malignant")
plt.tight_layout(); plt.savefig("fig_calib.pdf"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
w = .2
for i, (name, col) in enumerate([("LR", C1), ("SVM", C4), ("RF", C2), ("HGB", "#8064a2")]):
    ax[0].bar(np.arange(5) + (i - 1.5) * w, PI[name].mean(0), w, yerr=PI[name].std(0), color=col, label=name, error_kw=dict(lw=.6))
    ax[1].bar(np.arange(3) + (i - 1.5) * w, PG[name].mean(0), w, yerr=PG[name].std(0), color=col, label=name, error_kw=dict(lw=.6))
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(short); ax[0].set_ylabel("AUROC drop when permuted"); ax[0].set_title("(a) Single features", fontsize=9)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["size group", "texture", "smoothness"]); ax[1].set_title("(b) Correlated features permuted jointly", fontsize=9)
ax[0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig("fig_importance.pdf"); plt.close()

fig, ax = plt.subplots(figsize=(7.0, 2.2))
order = [("breast", 569, 0, 5), ("fourth source", 167497, 5, 9), ("lung", 23658, 9, 29), ("prostate", 100, 29, 38)]
M = np.zeros((4, 39))
for i, (_, _, a, b) in enumerate(order): M[i, a:b] = 1
M[1, 38] = 1; M[2, 38] = 1
ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#f0f0f0", C1]), aspect="auto")
ax.set_yticks(range(4)); ax.set_yticklabels([f"{n} (n = {k:,})" for n, k, _, _ in order]); ax.set_xticks([2, 7, 19, 33.5, 38]); ax.set_xticklabels(["5 cols", "4 cols", "20 cols", "9 cols", "Age"])
ax.set_xlabel("Columns of the fused 39-column table (dark = observed, light = structurally missing)")
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, 39, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True); ax.grid(which="minor", color="white", lw=.6); ax.tick_params(which="minor", length=0)
plt.tight_layout(); plt.savefig("fig_blocks.pdf"); plt.close()
print("done")
