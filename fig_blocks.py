"""Figure 3: block structure of the fused table, drawn from scaler_columns.csv."""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "font.family": "serif"})

rows = list(csv.DictReader(open("scaler_columns.csv")))
count = {r["column"]: int(r["n_observed"]) for r in rows}
size = {"breast": 569, "colorectal": 167497, "lung": 23658, "prostate": 100}
shared = [c for c, k in count.items() if k == size["lung"] + size["colorectal"]]
own = {b: [c for c, k in count.items() if k == n] for b, n in size.items()}
order = ["breast", "colorectal", "lung", "prostate"]
layout = [c for b in order for c in own[b]] + shared            # column order on the x axis
M = np.zeros((len(order), len(layout)))
for i, b in enumerate(order):
    for c in own[b]:
        M[i, layout.index(c)] = 1
    if b in ("lung", "colorectal"):
        for c in shared:
            M[i, layout.index(c)] = 1

fig, ax = plt.subplots(figsize=(7.0, 2.2))
ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#f0f0f0", "#1f4e79"]), aspect="auto")
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{b} (n = {size[b]:,})" for b in order])
ticks, labels, start = [], [], 0
for b in order:
    k = len(own[b]); ticks.append(start + (k - 1) / 2); labels.append(f"{k} cols"); start += k
ticks.append(start + (len(shared) - 1) / 2); labels.append("Age,\ntumour size")
ax.set_xticks(ticks); ax.set_xticklabels(labels)
ax.set_xlabel("Columns of the fused table (dark: observed; light: structurally missing)")
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, len(layout), 1), minor=True); ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
ax.grid(which="minor", color="white", lw=.6); ax.tick_params(which="minor", length=0)
plt.tight_layout(); plt.savefig("fig_blocks.pdf"); plt.close()
print({b: len(own[b]) for b in order}, shared)
