"""
Generate all figures for the Millikan Oil Drop lab report from raw data.
Reads the marks JSON files and charge_results CSVs, computes velocity
statistics, and produces publication-quality plots.
"""

import json
import math
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
e_KNOWN = 1.602176634e-19  # CODATA 2018 elementary charge (C)
RETICLE_DIST = 0.5e-3      # 0.500 mm per major division (m)

plt.rcParams.update({
    "figure.dpi": 300, "font.size": 10, "font.family": "serif",
    "lines.linewidth": 1.5, "axes.linewidth": 1.2,
})

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent  # figures go in repo root

# ── Load marks JSON and compute per-interval velocities ──────────────────────

def load_trial_velocities(json_path):
    """Return list of dicts, one per droplet, each with falls/rises lists."""
    with open(json_path) as f:
        data = json.load(f)
    fps = data["fps"]
    results = []
    for drop in data.get("droplets", []):
        marks = drop["marks"]
        falls, rises = [], []
        for i in range(len(marks) - 1):
            m0, m1 = marks[i], marks[i + 1]
            dt = abs(m1["time_s"] - m0["time_s"])
            if dt < 1e-6:
                continue
            v = RETICLE_DIST / dt  # m/s
            direction = f"{m0['type']}->{m1['type']}"
            entry = {"v": v, "dt": dt, "t_mid": (m0["time_s"] + m1["time_s"]) / 2.0,
                     "frame0": m0["frame"], "frame1": m1["frame"]}
            if direction == "top->bottom":
                falls.append(entry)
            elif direction == "bottom->top":
                rises.append(entry)
        results.append({
            "drop_id": drop["id"],
            "fps": fps,
            "falls": falls,
            "rises": rises,
            "v_fall_mean": np.mean([f["v"] for f in falls]) if falls else 0,
            "v_rise_mean": np.mean([r["v"] for r in rises]) if rises else 0,
            "v_fall_std": np.std([f["v"] for f in falls], ddof=1) if len(falls) > 1 else 0,
            "v_rise_std": np.std([r["v"] for r in rises], ddof=1) if len(rises) > 1 else 0,
            "n_fall": len(falls),
            "n_rise": len(rises),
        })
    return results

# ── Load CSV charge results ──────────────────────────────────────────────────

def load_charge_csv(csv_path):
    """Read a charge_results CSV and return list of row dicts."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

# ── Gather all data ──────────────────────────────────────────────────────────

trial_json = {
    1: DATA_DIR / "TRIAL_1_marks.json",
    2: DATA_DIR / "TRIAL_2_marks.json",
    3: DATA_DIR / "TRIAL_3_marks.json",
}
trial_csv = {
    1: DATA_DIR / "charge_results_TRIAL_1.csv",
    2: DATA_DIR / "charge_results.csv",          # Trial 2 results
    3: DATA_DIR / "charge_results_TRIAL_3.csv",
}

all_vel = {}   # trial -> [drop1_vel, drop2_vel]
all_chg = {}   # trial -> [drop1_row, drop2_row]
for t in (1, 2, 3):
    all_vel[t] = load_trial_velocities(trial_json[t])
    all_chg[t] = load_charge_csv(trial_csv[t])

# ── Print velocity tables (for LaTeX) ────────────────────────────────────────

print("=" * 80)
print("VELOCITY DATA FOR LATEX TABLES")
print("=" * 80)
for t in (1, 2, 3):
    print(f"\n--- Trial {t} ---")
    for d in all_vel[t]:
        state = "Pre-ionization" if d["drop_id"] == 1 else "Post-ionization"
        print(f"  {state} (Drop {d['drop_id']}):")
        print(f"    v_fall = {d['v_fall_mean']*1e3:.4f} mm/s  "
              f"± {d['v_fall_std']*1e3:.4f} mm/s  ({d['n_fall']} cycles)")
        print(f"    v_rise = {d['v_rise_mean']*1e3:.4f} mm/s  "
              f"± {d['v_rise_std']*1e3:.4f} mm/s  ({d['n_rise']} cycles)")
        print(f"    Individual falls (mm/s): "
              + ", ".join(f"{f['v']*1e3:.4f}" for f in d["falls"]))
        print(f"    Individual rises (mm/s): "
              + ", ".join(f"{r['v']*1e3:.4f}" for r in d["rises"]))

# ── Compile charge data ──────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("CHARGE RESULTS")
print("=" * 80)

drops = []  # flat list of all 6 drops
for t in (1, 2, 3):
    for row in all_chg[t]:
        q   = abs(float(row["q_C"]))
        dq  = float(row["delta_q_total"])
        dqV = float(row["delta_q_V"])
        dqT = float(row["delta_q_T"])
        dq_stat = float(row["delta_q_stat"])
        n   = int(row["n_est"])
        a1  = float(row["a1_m"])
        vf  = float(row["v_fall"])
        vr  = float(row["v_rise"])
        did = int(row["droplet_id"])
        drops.append({
            "trial": t, "drop_id": did,
            "state": "Pre" if did == 1 else "Post",
            "q": q, "dq": dq, "dqV": dqV, "dqT": dqT, "dq_stat": dq_stat,
            "n": n, "qe": q / e_KNOWN, "a1_um": a1 * 1e6,
            "v_fall": vf, "v_rise": vr,
        })
        print(f"  T{t}D{did}: q = {q:.4e} C  ± {dq:.2e}  n={n}  q/e={q/e_KNOWN:.3f}  "
              f"a1={a1*1e6:.3f} µm  δq_stat={dq_stat:.2e}  δq_V={dqV:.2e}  δq_T={dqT:.2e}")

# ── Elementary charge determination ──────────────────────────────────────────

e_ests = [d["q"] / d["n"] for d in drops]
e_errs = [d["dq"] / d["n"] for d in drops]
weights = np.array([1.0 / s**2 for s in e_errs])
e_weighted = float(np.average(e_ests, weights=weights))
e_weighted_err = 1.0 / math.sqrt(float(np.sum(weights)))
pct_err = abs(e_weighted - e_KNOWN) / e_KNOWN * 100

print(f"\n  e (weighted avg) = ({e_weighted:.4e} ± {e_weighted_err:.2e}) C")
print(f"  e (CODATA)       =  {e_KNOWN:.6e} C")
print(f"  Percent error    =  {pct_err:.2f}%")

# ── Ionization effect ────────────────────────────────────────────────────────

print(f"\n{'='*80}\nIONIZATION EFFECT\n{'='*80}")
for t in (1, 2, 3):
    pre  = [d for d in drops if d["trial"] == t and d["drop_id"] == 1][0]
    post = [d for d in drops if d["trial"] == t and d["drop_id"] == 2][0]
    dq = post["q"] - pre["q"]
    dn = post["n"] - pre["n"]
    print(f"  Trial {t}: q_pre={pre['q']:.3e}(n={pre['n']}), "
          f"q_post={post['q']:.3e}(n={post['n']}), "
          f"Δq={dq:+.3e} C, Δn={dn:+d}")

# ── Uncertainty budget (average fractional) ──────────────────────────────────

print(f"\n{'='*80}\nUNCERTAINTY BUDGET (fractional)\n{'='*80}")
for d in drops:
    frac_stat = d["dq_stat"] / d["q"] * 100
    frac_V    = d["dqV"] / d["q"] * 100
    frac_T    = d["dqT"] / d["q"] * 100
    frac_tot  = d["dq"] / d["q"] * 100
    print(f"  T{d['trial']}D{d['drop_id']}: stat={frac_stat:.1f}%  "
          f"V={frac_V:.2f}%  T={frac_T:.3f}%  total={frac_tot:.1f}%")

avg_stat = np.mean([d["dq_stat"] / d["q"] for d in drops]) * 100
avg_V    = np.mean([d["dqV"] / d["q"] for d in drops]) * 100
avg_T    = np.mean([d["dqT"] / d["q"] for d in drops]) * 100
print(f"\n  Average: stat={avg_stat:.1f}%  V={avg_V:.2f}%  T={avg_T:.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Charge Quantization ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
x_labels = [f"T{d['trial']}D{d['drop_id']}\n({d['state']})" for d in drops]
qe_vals  = [d["qe"] for d in drops]
qe_errs  = [d["dq"] / e_KNOWN for d in drops]
colors   = ["steelblue" if d["drop_id"] == 1 else "coral" for d in drops]
markers  = ["o" if d["drop_id"] == 1 else "s" for d in drops]

for i, (x, qe, err, c, m) in enumerate(zip(range(len(drops)), qe_vals, qe_errs, colors, markers)):
    ax.errorbar(x, qe, yerr=err, fmt=m, color=c, markersize=8,
                capsize=5, capthick=1.5, elinewidth=1.5, zorder=3)

# Integer reference lines
n_min, n_max = min(d["n"] for d in drops), max(d["n"] for d in drops)
for n in range(n_min - 1, n_max + 2):
    ls = "--" if n in {d["n"] for d in drops} else ":"
    lw = 1.5 if n in {d["n"] for d in drops} else 0.6
    al = 0.7 if n in {d["n"] for d in drops} else 0.3
    ax.axhline(n, color="gray", linestyle=ls, linewidth=lw, alpha=al)
    if n_min - 1 <= n <= n_max + 1:
        ax.text(len(drops) - 0.3, n + 0.15, f"$n={n}$", fontsize=8, color="gray")

pre_patch  = mpatches.Patch(color="steelblue", label="Pre-ionization")
post_patch = mpatches.Patch(color="coral",      label="Post-ionization")
ax.legend(handles=[pre_patch, post_patch], loc="upper left", fontsize=10)
ax.set_xticks(range(len(drops)))
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Charge ($q/e$)", fontsize=11)
ax.set_title("Charge Quantization: Measured Charges in Units of $e$", fontsize=12, fontweight="bold")
ax.set_ylim(n_min - 1.5, n_max + 1.5)
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "charge_quantization.png", dpi=300, bbox_inches="tight")
plt.close()
print("\n✓ charge_quantization.png")

# ── Figure 2: Ionization Effect ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
trials_list = [1, 2, 3]
x = np.arange(len(trials_list))
width = 0.35

pre_q  = [next(d["q"] * 1e19 for d in drops if d["trial"] == t and d["drop_id"] == 1) for t in trials_list]
post_q = [next(d["q"] * 1e19 for d in drops if d["trial"] == t and d["drop_id"] == 2) for t in trials_list]
pre_n  = [next(d["n"] for d in drops if d["trial"] == t and d["drop_id"] == 1) for t in trials_list]
post_n = [next(d["n"] for d in drops if d["trial"] == t and d["drop_id"] == 2) for t in trials_list]

bars1 = ax.bar(x - width/2, pre_q, width, label="Pre-ionization",
               color="steelblue", alpha=0.8, edgecolor="black", linewidth=1.2)
bars2 = ax.bar(x + width/2, post_q, width, label="Post-ionization",
               color="coral", alpha=0.8, edgecolor="black", linewidth=1.2)

for bars, n_vals in [(bars1, pre_n), (bars2, post_n)]:
    for bar, n in zip(bars, n_vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.2,
                f"$n={n}$", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Annotate Δn
for i, t in enumerate(trials_list):
    dn = post_n[i] - pre_n[i]
    sign = "+" if dn > 0 else ""
    y_arr = max(pre_q[i], post_q[i]) + 1.5
    ax.annotate(f"$\\Delta n = {sign}{dn}$",
                xy=(i, y_arr), fontsize=10, ha="center", color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel("Charge ($\\times 10^{-19}$ C)", fontsize=11)
ax.set_title("Ionization-Induced Charge Changes", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Trial {t}" for t in trials_list])
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, max(max(pre_q), max(post_q)) + 4)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "ionization_effect.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ ionization_effect.png")

# ── Figure 3: Error Budget ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5.5))

# Average variance contributions across all drops (in quadrature, so use variance fractions)
var_stat = np.mean([(d["dq_stat"]**2) / (d["dq"]**2) for d in drops]) * 100
var_V    = np.mean([(d["dqV"]**2) / (d["dq"]**2) for d in drops]) * 100
var_T    = np.mean([(d["dqT"]**2) / (d["dq"]**2) for d in drops]) * 100

labels = [f"Velocity scatter\n(statistical)\n{var_stat:.1f}%",
          f"Applied voltage\n(systematic)\n{var_V:.1f}%",
          f"Temperature\n(systematic)\n{var_T:.1f}%"]
sizes  = [var_stat, var_V, var_T]
colors_pie = ["#FF6B6B", "#4ECDC4", "#95E1D3"]
explode = (0.05, 0, 0)

wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct="%1.1f%%", colors=colors_pie,
    explode=explode, startangle=90, textprops={"fontsize": 10})
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
    at.set_fontsize(11)

ax.set_title("Uncertainty Budget: Variance Contributions\n(velocity-limited measurement)",
             fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(OUT_DIR / "error_budget_pie.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ error_budget_pie.png")

# ── Figure 4: Result Comparison ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
bar_width = 0.4

ax.bar(0, e_weighted * 1e19, bar_width,
       color="steelblue", alpha=0.7, edgecolor="black", linewidth=1.5,
       label="Experimental")
ax.errorbar(0, e_weighted * 1e19, yerr=e_weighted_err * 1e19,
            fmt="none", ecolor="steelblue", elinewidth=3, capsize=8, capthick=2)
ax.axhline(e_KNOWN * 1e19, color="red", linestyle="-", linewidth=2.5,
           label="CODATA 2018", zorder=3)

ax.text(0, (e_weighted + e_weighted_err) * 1e19 + 0.003,
        f"${e_weighted*1e19:.3f} \\pm {e_weighted_err*1e19:.3f}$"
        f"$\\times 10^{{-19}}$ C",
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.text(0.25, e_KNOWN * 1e19 + 0.002,
        f"Agreement:\n{pct_err:.1f}%", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

ax.set_ylabel("Elementary Charge ($\\times 10^{-19}$ C)", fontsize=11)
ax.set_title("Experimental Result vs. Accepted Value", fontsize=12, fontweight="bold")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(1.50, 1.72)
ax.set_xticks([])
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "result_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ result_comparison.png")

# ── Figure 5: Per-drop e estimate ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = range(len(drops))
e_vals_19 = [est * 1e19 for est in e_ests]
e_errs_19 = [err * 1e19 for err in e_errs]

for i, (xp, ev, ee, d) in enumerate(zip(x_pos, e_vals_19, e_errs_19, drops)):
    c = "steelblue" if d["drop_id"] == 1 else "coral"
    m = "o" if d["drop_id"] == 1 else "s"
    ax.errorbar(xp, ev, yerr=ee, fmt=m, color=c, markersize=8,
                capsize=5, capthick=1.5, elinewidth=1.5, zorder=3)

ax.axhline(e_KNOWN * 1e19, color="red", linewidth=2, label="CODATA 2018")
ax.axhspan((e_weighted - e_weighted_err) * 1e19,
           (e_weighted + e_weighted_err) * 1e19,
           alpha=0.15, color="steelblue", label=f"Weighted avg ± 1$\\sigma$")

ax.set_xticks(list(x_pos))
ax.set_xticklabels([f"T{d['trial']}D{d['drop_id']}" for d in drops], fontsize=9)
ax.set_ylabel("$q/n$ ($\\times 10^{-19}$ C)", fontsize=11)
ax.set_title("Per-Droplet Elementary Charge Estimates ($q/n$)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "per_drop_e_estimate.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ per_drop_e_estimate.png")

print(f"\nAll figures generated successfully.")
