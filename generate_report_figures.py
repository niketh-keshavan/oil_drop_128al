"""
Generate all figures for the Millikan Oil Drop lab report from raw data.
Reads the marks JSON files and charge_results CSVs, computes velocity
statistics, and produces publication-quality plots.

Includes per-cycle charge analysis to detect and account for mid-oscillation
charge changes (stray ionization events between cycles).
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
g       = 9.80665           # m/s²
RHO_OIL = 853.0             # kg/m³
b_CUNN  = 8.226e-3          # Cunningham constant (Pa·m)
P_ATM   = 1.01325e5         # Pa
V_NOM   = 553.5             # V (nominal applied voltage)
D_SEP   = 7.59e-3           # m (plate separation — from calculate_charge.py interactive default)

# Air properties at T_nom = 22.35 °C
T_NOM   = 22.35             # °C
ETA_NOM = 1.828e-5 + 0.35 * (1.832e-5 - 1.828e-5)  # interpolated η at 22.35 °C
M_AIR   = 0.02897           # kg/mol
R_GAS   = 8.314             # J/(mol·K)
RHO_AIR = P_ATM * M_AIR / (R_GAS * (T_NOM + 273.15))

plt.rcParams.update({
    "figure.dpi": 300, "font.size": 10, "font.family": "serif",
    "lines.linewidth": 1.5, "axes.linewidth": 1.2,
})

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent  # figures go in repo root

# ── Charge calculation from physics ──────────────────────────────────────────

def compute_charge_from_velocities(v_fall, v_rise, voltage=V_NOM, d_sep=D_SEP,
                                    eta=ETA_NOM, rho_oil=RHO_OIL, rho_air=RHO_AIR,
                                    pressure=P_ATM):
    """Compute Cunningham-corrected charge from fall/rise terminal velocities."""
    drho = rho_oil - rho_air
    a1 = math.sqrt(9.0 * eta * v_fall / (2.0 * drho * g))
    coeff = (18.0 * math.pi * d_sep) / voltage
    q0 = coeff * math.sqrt(eta**3 * v_fall / (2.0 * drho * g)) * (v_rise + v_fall)
    corr = 1.0 + b_CUNN / (pressure * a1)
    q = q0 / corr**1.5
    return {"q": q, "a1": a1, "q0": q0, "corr": corr, "n_est": round(abs(q) / e_KNOWN)}

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
                     "frame0": m0["frame"], "frame1": m1["frame"], "idx": i}
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

# ── Per-cycle charge computation ─────────────────────────────────────────────

def compute_per_cycle_charges(drop_vel):
    """For each rise cycle, compute charge using mean v_fall and that cycle's v_rise.

    Since v_fall depends only on mass (constant), we use the mean fall velocity
    to determine the droplet radius, then compute a per-cycle charge for each
    rise velocity. This allows detection of charge changes during the oscillation
    period.
    """
    v_fall = drop_vel["v_fall_mean"]
    per_cycle = []
    for r in drop_vel["rises"]:
        res = compute_charge_from_velocities(v_fall, r["v"])
        per_cycle.append({
            "t_mid": r["t_mid"],
            "v_rise": r["v"],
            "v_fall": v_fall,
            "q": abs(res["q"]),
            "n_est": res["n_est"],
            "qe": abs(res["q"]) / e_KNOWN,
            "a1": res["a1"],
        })
    return per_cycle

def identify_charge_states(per_cycle, threshold_frac=0.15):
    """Group consecutive cycles into charge states using step detection.

    A new charge state is identified when consecutive rise velocities differ
    by more than threshold_frac (fraction of the smaller value).
    Returns list of state dicts with mean q, n, and member cycle indices.
    """
    if not per_cycle:
        return []

    states = []
    current = [0]  # indices into per_cycle

    for i in range(1, len(per_cycle)):
        v_prev = per_cycle[i-1]["v_rise"]
        v_curr = per_cycle[i]["v_rise"]
        frac_change = abs(v_curr - v_prev) / min(v_prev, v_curr)
        if frac_change > threshold_frac:
            # Close current state
            states.append(current)
            current = [i]
        else:
            current.append(i)
    states.append(current)

    result = []
    for s_indices in states:
        cycles = [per_cycle[i] for i in s_indices]
        qs = [c["q"] for c in cycles]
        vrs = [c["v_rise"] for c in cycles]
        mean_q = np.mean(qs)
        n_est = round(mean_q / e_KNOWN)
        result.append({
            "indices": s_indices,
            "cycles": cycles,
            "mean_q": mean_q,
            "std_q": np.std(qs, ddof=1) if len(qs) > 1 else 0,
            "mean_vr": np.mean(vrs),
            "n_est": n_est,
            "qe": mean_q / e_KNOWN,
            "n_cycles": len(s_indices),
            "t_start": cycles[0]["t_mid"],
            "t_end": cycles[-1]["t_mid"],
        })
    return result

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

# ── Per-cycle charge analysis ────────────────────────────────────────────────

print("=" * 80)
print("PER-CYCLE CHARGE ANALYSIS")
print("=" * 80)

all_per_cycle = {}   # (trial, drop_id) -> per_cycle list
all_states    = {}   # (trial, drop_id) -> charge states
all_transitions = [] # list of (Δn, Δq) for charge state transitions

for t in (1, 2, 3):
    for d in all_vel[t]:
        key = (t, d["drop_id"])
        pc = compute_per_cycle_charges(d)
        all_per_cycle[key] = pc
        states = identify_charge_states(pc)
        all_states[key] = states

        state_label = "Pre-ion" if d["drop_id"] == 1 else "Post-ion"
        print(f"\n  T{t}D{d['drop_id']} ({state_label}):")
        print(f"    v_fall(mean) = {d['v_fall_mean']*1e3:.4f} mm/s  (CV={d['v_fall_std']/d['v_fall_mean']*100:.1f}%)")
        print(f"    Per-cycle charges (q/e):")
        for i, c in enumerate(pc):
            print(f"      cycle {i}: v_r={c['v_rise']*1e3:.4f} mm/s  "
                  f"q={c['q']:.3e} C  q/e={c['qe']:.2f}  n≈{c['n_est']}")

        print(f"    Identified {len(states)} charge state(s):")
        for j, st in enumerate(states):
            print(f"      State {j}: n≈{st['n_est']} (q/e={st['qe']:.2f})  "
                  f"{st['n_cycles']} cycle(s)  "
                  f"q={st['mean_q']:.3e}±{st['std_q']:.1e} C")

        # Record transitions between consecutive charge states
        for j in range(1, len(states)):
            dq = states[j]["mean_q"] - states[j-1]["mean_q"]
            dn = states[j]["n_est"] - states[j-1]["n_est"]
            all_transitions.append({
                "trial": t, "drop_id": d["drop_id"],
                "from_n": states[j-1]["n_est"], "to_n": states[j]["n_est"],
                "dn": dn, "dq": dq,
            })
            print(f"      Transition: n={states[j-1]['n_est']} → {states[j]['n_est']}  "
                  f"(Δn={dn:+d}, Δq={dq:+.3e} C)")

# ── Compile charge data from CSVs (original mean-based analysis) ─────────────

print("\n" + "=" * 80)
print("ORIGINAL MEAN-BASED CHARGE RESULTS (from CSVs)")
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
        print(f"  T{t}D{did}: q = {q:.4e} C  ± {dq:.2e}  n={n}  q/e={q/e_KNOWN:.3f}")

# ── Elementary charge from CHARGE STATES (per-cycle analysis) ─────────────────

print("\n" + "=" * 80)
print("ELEMENTARY CHARGE FROM CHARGE STATES")
print("=" * 80)

# Collect all charge states across all droplets
all_state_list = []
for t in (1, 2, 3):
    for d in all_vel[t]:
        key = (t, d["drop_id"])
        for st in all_states[key]:
            all_state_list.append({
                "trial": t, "drop_id": d["drop_id"],
                "n": st["n_est"], "q": st["mean_q"], "dq": st["std_q"],
                "n_cycles": st["n_cycles"], "qe": st["qe"],
            })
            print(f"  T{t}D{d['drop_id']}: state n≈{st['n_est']}  "
                  f"q={st['mean_q']:.4e} C  ({st['n_cycles']} cycles)")

# e from q/n for each state (weighted by 1/σ²; use dq if available, else rough estimate)
e_state_ests = []
e_state_errs = []
for s in all_state_list:
    e_est = s["q"] / s["n"]
    # Estimate uncertainty: if std available use it, else use 10% of q
    dq = s["dq"] if s["dq"] > 0 else 0.10 * s["q"]
    e_err = dq / s["n"]
    e_state_ests.append(e_est)
    e_state_errs.append(e_err)

w = np.array([1.0 / s**2 if s > 0 else 0 for s in e_state_errs])
e_states_weighted = float(np.average(e_state_ests, weights=w))
e_states_err = 1.0 / math.sqrt(float(np.sum(w)))
pct_states = abs(e_states_weighted - e_KNOWN) / e_KNOWN * 100

print(f"\n  e (charge-state weighted avg) = ({e_states_weighted:.4e} ± {e_states_err:.2e}) C")
print(f"  CODATA                        = {e_KNOWN:.6e} C")
print(f"  Percent error                 = {pct_states:.2f}%")

# ── e from transition differences ────────────────────────────────────────────

print(f"\n{'='*80}\ne FROM CHARGE STATE TRANSITIONS (Δq/Δn)\n{'='*80}")
e_trans_ests = []
for tr in all_transitions:
    if tr["dn"] != 0:
        e_tr = abs(tr["dq"]) / abs(tr["dn"])
        e_trans_ests.append(e_tr)
        print(f"  T{tr['trial']}D{tr['drop_id']}: n {tr['from_n']}→{tr['to_n']}  "
              f"|Δq|/|Δn| = {e_tr:.4e} C  (= {e_tr/e_KNOWN:.3f} e)")

if e_trans_ests:
    e_trans_mean = np.mean(e_trans_ests)
    e_trans_std  = np.std(e_trans_ests, ddof=1) if len(e_trans_ests) > 1 else 0
    pct_trans = abs(e_trans_mean - e_KNOWN) / e_KNOWN * 100
    print(f"\n  e (from transitions) = ({e_trans_mean:.4e} ± {e_trans_std:.2e}) C")
    print(f"  Percent error        = {pct_trans:.2f}%")

# ── Use original mean-based for overall result (backward compat) ─────────────

e_ests = [d["q"] / d["n"] for d in drops]
e_errs = [d["dq"] / d["n"] for d in drops]
weights_orig = np.array([1.0 / s**2 for s in e_errs])
e_weighted = float(np.average(e_ests, weights=weights_orig))
e_weighted_err = 1.0 / math.sqrt(float(np.sum(weights_orig)))
pct_err = abs(e_weighted - e_KNOWN) / e_KNOWN * 100

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

# ── Figure: Per-Cycle Charge Time Series ─────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
fig.suptitle("Per-Cycle Charge Analysis: Detecting Mid-Oscillation Charge Changes",
             fontsize=13, fontweight="bold", y=0.98)

for row_idx, t in enumerate((1, 2, 3)):
    for col_idx, d in enumerate(all_vel[t]):
        ax = axes[row_idx, col_idx]
        key = (t, d["drop_id"])
        pc = all_per_cycle[key]
        states = all_states[key]
        state_label = "Pre-ionization" if d["drop_id"] == 1 else "Post-ionization"

        # Plot individual cycle q/e values
        times = [c["t_mid"] for c in pc]
        qe_vals = [c["qe"] for c in pc]
        ax.plot(times, qe_vals, "ko", markersize=6, zorder=5, label="Per-cycle $q/e$")

        # Shade charge state regions
        cmap = ["#AED6F1", "#ABEBC6", "#F9E79F", "#F5B7B1", "#D7BDE2"]
        for j, st in enumerate(states):
            t_start = st["cycles"][0]["t_mid"] - 1.0
            t_end   = st["cycles"][-1]["t_mid"] + 1.0
            color = cmap[j % len(cmap)]
            ax.axhspan(st["n_est"] - 0.5, st["n_est"] + 0.5,
                       xmin=0, xmax=1, alpha=0.15, color=color)
            ax.axhline(st["n_est"], color="gray", ls="--", lw=1, alpha=0.6)
            # Label state
            ax.text(st["cycles"][0]["t_mid"], st["n_est"] + 0.35,
                    f"$n={st['n_est']}$", fontsize=9, color="darkblue",
                    fontweight="bold", ha="center")

        # Mark transitions with vertical lines
        for j in range(1, len(states)):
            t_trans = (states[j-1]["cycles"][-1]["t_mid"] +
                       states[j]["cycles"][0]["t_mid"]) / 2
            ax.axvline(t_trans, color="red", ls=":", lw=1.5, alpha=0.7)
            dn = states[j]["n_est"] - states[j-1]["n_est"]
            sign = "+" if dn > 0 else ""
            ax.text(t_trans, max(qe_vals) + 0.3, f"$\\Delta n={sign}{dn}$",
                    fontsize=8, color="red", ha="center", rotation=0)

        ax.set_title(f"Trial {t}, Drop {d['drop_id']} ({state_label})", fontsize=10)
        ax.set_ylabel("$q/e$", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Set y-limits based on data range
        ymin = min(qe_vals) - 1.5
        ymax = max(qe_vals) + 1.5
        ax.set_ylim(ymin, ymax)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_DIR / "per_cycle_charge_timeseries.png", dpi=300, bbox_inches="tight")
plt.close()
print("\n✓ per_cycle_charge_timeseries.png")

# ── Figure 1: Charge Quantization (updated with charge states) ───────────────

fig, ax = plt.subplots(figsize=(10, 5.5))

# Plot all individual charge states (not just mean per droplet)
x_pos = 0
x_labels_cs = []
qe_vals_cs = []
qe_errs_cs = []
colors_cs  = []
markers_cs = []
n_vals_cs  = []

for t in (1, 2, 3):
    for d in all_vel[t]:
        key = (t, d["drop_id"])
        for st in all_states[key]:
            x_labels_cs.append(f"T{t}D{d['drop_id']}\nn={st['n_est']}")
            qe_vals_cs.append(st["qe"])
            # Error: use std if > 1 cycle, else rough 10% estimate
            err_q = st["std_q"] if st["std_q"] > 0 else 0.10 * st["mean_q"]
            qe_errs_cs.append(err_q / e_KNOWN)
            colors_cs.append("steelblue" if d["drop_id"] == 1 else "coral")
            markers_cs.append("o" if d["drop_id"] == 1 else "s")
            n_vals_cs.append(st["n_est"])

for i in range(len(qe_vals_cs)):
    ax.errorbar(i, qe_vals_cs[i], yerr=qe_errs_cs[i], fmt=markers_cs[i],
                color=colors_cs[i], markersize=8,
                capsize=5, capthick=1.5, elinewidth=1.5, zorder=3)

# Integer reference lines
n_set = set(n_vals_cs)
n_min, n_max = min(n_vals_cs), max(n_vals_cs)
for n in range(n_min - 1, n_max + 2):
    ls = "--" if n in n_set else ":"
    lw = 1.5 if n in n_set else 0.6
    al = 0.7 if n in n_set else 0.3
    ax.axhline(n, color="gray", linestyle=ls, linewidth=lw, alpha=al)
    ax.text(len(qe_vals_cs) - 0.3, n + 0.15, f"$n={n}$", fontsize=8, color="gray")

pre_patch  = mpatches.Patch(color="steelblue", label="Pre-ionization")
post_patch = mpatches.Patch(color="coral",      label="Post-ionization")
ax.legend(handles=[pre_patch, post_patch], loc="upper left", fontsize=10)
ax.set_xticks(range(len(qe_vals_cs)))
ax.set_xticklabels(x_labels_cs, fontsize=8)
ax.set_ylabel("Charge ($q/e$)", fontsize=11)
ax.set_title("Charge Quantization: All Identified Charge States",
             fontsize=12, fontweight="bold")
ax.set_ylim(n_min - 1.5, n_max + 1.5)
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "charge_quantization.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ charge_quantization.png")

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
bar_width = 0.35

ax.bar(-0.2, e_weighted * 1e19, bar_width,
       color="steelblue", alpha=0.7, edgecolor="black", linewidth=1.5,
       label="Mean-based")
ax.errorbar(-0.2, e_weighted * 1e19, yerr=e_weighted_err * 1e19,
            fmt="none", ecolor="steelblue", elinewidth=3, capsize=8, capthick=2)
ax.bar(0.2, e_states_weighted * 1e19, bar_width,
       color="coral", alpha=0.7, edgecolor="black", linewidth=1.5,
       label="Charge-state")
ax.errorbar(0.2, e_states_weighted * 1e19, yerr=e_states_err * 1e19,
            fmt="none", ecolor="coral", elinewidth=3, capsize=8, capthick=2)
ax.axhline(e_KNOWN * 1e19, color="red", linestyle="-", linewidth=2.5,
           label="CODATA 2018", zorder=3)

ax.text(-0.2, (e_weighted + e_weighted_err) * 1e19 + 0.003,
        f"${e_weighted*1e19:.3f} \\pm {e_weighted_err*1e19:.3f}$",
        ha="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax.text(0.2, (e_states_weighted + e_states_err) * 1e19 + 0.003,
        f"${e_states_weighted*1e19:.3f} \\pm {e_states_err*1e19:.3f}$",
        ha="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.5))

ax.set_ylabel("Elementary Charge ($\\times 10^{-19}$ C)", fontsize=11)
ax.set_title("Experimental Result vs. Accepted Value", fontsize=12, fontweight="bold")
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(1.45, 1.80)
ax.set_xticks([-0.2, 0.2])
ax.set_xticklabels(["Mean-based", "Charge-state"], fontsize=10)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "result_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ result_comparison.png")

# ── Figure 5: Per-charge-state e estimate ────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = 0
xtick_pos = []
xtick_lbl = []
colors_state = {"steelblue": 0, "coral": 0}  # just for legend

for t in (1, 2, 3):
    for d in all_vel[t]:
        key = (t, d["drop_id"])
        c = "steelblue" if d["drop_id"] == 1 else "coral"
        m = "o" if d["drop_id"] == 1 else "s"
        for st in all_states[key]:
            e_est = st["mean_q"] / st["n_est"]
            dq = st["std_q"] if st["std_q"] > 0 else 0.10 * st["mean_q"]
            e_err = dq / st["n_est"]
            ax.errorbar(x_pos, e_est * 1e19, yerr=e_err * 1e19,
                        fmt=m, color=c, markersize=8,
                        capsize=5, capthick=1.5, elinewidth=1.5, zorder=3)
            xtick_pos.append(x_pos)
            xtick_lbl.append(f"T{t}D{d['drop_id']}\nn={st['n_est']}")
            x_pos += 1
    # small gap between trials
    x_pos += 0.5

ax.axhline(e_KNOWN * 1e19, color="red", linewidth=2, label="CODATA 2018")
ax.axhspan((e_states_weighted - e_states_err) * 1e19,
           (e_states_weighted + e_states_err) * 1e19,
           alpha=0.15, color="steelblue", label=f"Weighted avg ± 1$\\sigma$")

pre_patch  = mpatches.Patch(color="steelblue", label="Pre-ionization")
post_patch = mpatches.Patch(color="coral",      label="Post-ionization")
ax.legend(handles=[pre_patch, post_patch,
                   plt.Line2D([], [], color="red", lw=2, label="CODATA 2018"),
                   mpatches.Patch(alpha=0.15, color="steelblue",
                                  label=f"Weighted avg ± 1σ")],
          fontsize=9, loc="upper right")
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lbl, fontsize=8)
ax.set_ylabel("$q/n$ ($\\times 10^{-19}$ C)", fontsize=11)
ax.set_title("Per-Charge-State Elementary Charge Estimates ($q/n$)",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "per_drop_e_estimate.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ per_drop_e_estimate.png")

print(f"\nAll figures generated successfully.")
