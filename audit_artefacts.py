"""Inspection of the two surviving artefacts of the prototype (Sections 5.1 and 5.4).

Inputs (in artefacts/):
  solid_symptom_scaler.pkl        fitted StandardScaler (joblib pickle)
  multimodal_cancer_predictor.h5  saved Keras model

Outputs:
  results_artefacts.json   every number quoted in the text
  scaler_columns.csv       the 39 columns with count, mean and SD

Requires numpy, scikit-learn, joblib, h5py and keras/tensorflow.
"""
import csv, json, math, os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np, joblib, h5py

SCALER = "artefacts/solid_symptom_scaler.pkl"
MODEL = "artefacts/multimodal_cancer_predictor.h5"
R = {}

# ---------------- scaler ----------------
sc = joblib.load(SCALER)
cols = [str(c) for c in sc.feature_names_in_]
n = [int(v) for v in sc.n_samples_seen_]
mean = [float(v) for v in sc.mean_]
sd = [float(v) for v in sc.scale_]
with open("scaler_columns.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["column", "n_observed", "mean", "sd"])
    for row in zip(cols, n, mean, sd):
        w.writerow(row)

groups = {}
for c, k in zip(cols, n):
    groups.setdefault(k, []).append(c)
R["scaler"] = {
    "n_columns": len(cols),
    "columns_by_count": {str(k): v for k, v in sorted(groups.items())},
    "n_columns_by_count": {str(k): len(v) for k, v in sorted(groups.items())},
}
# Reading equal counts as shared rows: four blocks, two columns shared by the two large ones.
blocks = {"breast": 569, "prostate": 100, "lung": 23658, "colorectal": 167497}
total = sum(blocks.values())
R["scaler"]["blocks"] = blocks
R["scaler"]["total_rows"] = total
R["scaler"]["shares"] = {k: v / total for k, v in blocks.items()}
R["scaler"]["shared_count_check"] = {"lung_plus_colorectal": blocks["lung"] + blocks["colorectal"],
                                     "count_of_Age": n[cols.index("Age")]}
three = blocks["breast"] + blocks["lung"] + blocks["prostate"]
R["three_source_table"] = {"rows": three, "shares": {k: blocks[k] / three for k in ("breast", "lung", "prostate")}}

# Bounds implied by a uniform distribution with the same mean and SD (mean -/+ sqrt(3) SD).
R["scaler"]["uniform_implied_bounds"] = {
    c: [m - math.sqrt(3) * s, m + math.sqrt(3) * s]
    for c, k, m, s in zip(cols, n, mean, sd) if k in (23658, 167497)
}

# ---------------- model file ----------------
f = h5py.File(MODEL, "r")
cfg = json.loads(f.attrs["model_config"])
R["model"] = {"keras_version_saved": str(f.attrs["keras_version"]),
              "layers": [(L["class_name"], L["config"]["name"]) for L in cfg["config"]["layers"]]}
fan = {"dense_2": (3, 128), "dense_3": (128, 64), "dense_4": (1280, 256),
       "dense_5": (256, 128), "dense_6": (192, 128), "dense_7": (128, 3)}
opt = f["optimizer_weights"]["adam"]
lay = {}
for name, (fi, fo) in fan.items():
    k = np.array(f["model_weights"][name][name]["kernel"]); b = np.array(f["model_weights"][name][name]["bias"])
    km = np.array(opt[name + "_kernel_momentum"]); kv = np.array(opt[name + "_kernel_velocity"])
    bm = np.array(opt[name + "_bias_momentum"]); bv = np.array(opt[name + "_bias_velocity"])
    fin = k[np.isfinite(k)]
    lay[name] = {
        "glorot_uniform_limit": math.sqrt(6 / (fi + fo)),
        "kernel_max_abs": float(np.abs(fin).max()),
        "kernel_sd": float(fin.std()),
        "uniform_sd_for_limit": math.sqrt(6 / (fi + fo)) / math.sqrt(3),
        "kernel_nan": int(np.isnan(k).sum()),
        "kernel_nan_per_input_row": [int(v) for v in np.isnan(k).sum(1)] if name == "dense_2" else None,
        "bias_all_zero": bool((b == 0).all()),
        "adam_kernel_moments_all_zero": bool((km[np.isfinite(km)] == 0).all() and (kv[np.isfinite(kv)] == 0).all()),
        "adam_bias_moments_all_zero": bool((bm == 0).all() and (bv == 0).all()),
    }
R["model"]["dense_layers"] = lay
bias = np.array(f["model_weights"]["dense_7"]["dense_7"]["bias"], dtype=float)
p = np.exp(bias - bias.max()); p /= p.sum()
R["model"]["output_bias"] = [float(v) for v in bias]
R["model"]["softmax_of_output_bias"] = [float(v) for v in p]
it = int(np.array(opt["iteration"]))
R["model"]["adam_updates"] = it
# Which (epochs, batch size) reproduce the update count on 80% of the three-source table?
fits = []
for frac in (0.8,):
    for ntrain in {math.floor(frac * three), math.ceil(frac * three)}:
        for batch in (8, 16, 32, 64, 128):
            steps = math.ceil(ntrain / batch)
            if it % steps == 0:
                fits.append({"train_rows": ntrain, "batch_size": batch, "steps_per_epoch": steps, "epochs": it // steps})
R["model"]["update_count_fits_80pct_of_three_source_table"] = fits
f.close()

# ---------------- querying the model ----------------
import keras, tensorflow as tf
m = keras.models.load_model(MODEL, compile=False)
rng = np.random.default_rng(0)
S = rng.uniform(0, 1, (5000, 3)).astype("float32")
zero = np.zeros((5000, 1280), "float32")
Pz = m.predict([S, zero], verbose=0, batch_size=512)
img = np.clip(rng.exponential(0.4, (5000, 1280)), 0, 6).astype("float32")     # non-negative, as pooled ReLU6 features
Pi = m.predict([S, img], verbose=0, batch_size=512)
fixed = np.tile(img[:1], (2000, 1))
Pf = m.predict([S[:2000], fixed], verbose=0, batch_size=512)
R["queries"] = {
    "keras_version_used": keras.__version__, "tensorflow_version_used": tf.__version__,
    "zero_image_5000_score_triples": {"distinct_outputs": int(len(np.unique(Pz, axis=0))), "output": [float(v) for v in Pz[0]]},
    "fixed_image_2000_score_triples": {"max_abs_difference_between_outputs": float(np.abs(Pf - Pf[0]).max())},
    "random_images_5000": {"top_class_counts": [int(v) for v in np.bincount(Pi.argmax(1), minlength=3)],
                           "top_probability_min": float(Pi.max(1).min()), "top_probability_median": float(np.median(Pi.max(1))),
                           "top_probability_max": float(Pi.max(1).max()), "any_nan": bool(np.isnan(Pi).any())},
}
json.dump(R, open("results_artefacts.json", "w"), indent=1)
print(json.dumps({k: R[k] for k in ("three_source_table", "queries")}, indent=1))
print("softmax of output bias:", R["model"]["softmax_of_output_bias"], "| updates:", it, "| fits:", fits)
