"""
Millikan Oil Drop Experiment — Charge Calculator (with uncertainties)
=====================================================================
Reads the oscillation marks produced by  video_player.py  and computes the
electric charge on each oil droplet using the formulas from the UCSB Physics
128AL lab manual (PASCO AP-8210A apparatus).

Temperature is derived from 10 kΩ NTC thermistor resistance readings.
Voltage and thermistor readings are given as low/high bounds so that
systematic uncertainties are propagated through the calculation.

Key equations (see manual §3 Theory)
-------------------------------------
  (5)  a₁  = sqrt( 9·η·v_f / (2·(ρ − ρ_air)·g) )
  (6)  q₀  = (18·π·d / V) · sqrt( η³·v_f / (2·(ρ − ρ_air)·g) ) · (v_r + v_f)
  (10) q   = q₀ / [ 1 + b/(p·a₁) ]^(3/2)           (Cunningham correction)

Usage
-----
    python calculate_charge.py                          # interactive
    python calculate_charge.py data/TRIAL_1_marks.json  # specific file
"""

import json
import sys
import math
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no Tk needed
import matplotlib.pyplot as plt
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
#  Physical constants and apparatus parameters
# ═══════════════════════════════════════════════════════════════════════════════

g = 9.80665                 # gravitational acceleration  (m/s²)
RHO_OIL = 853.0             # mineral oil density (kg/m³) @20 °C  (manual: 853 ± 2)
b_CUNNINGHAM = 8.226e-3     # Cunningham correction constant  (Pa·m)
P_ATM_DEFAULT = 1.01325e5   # standard atmospheric pressure  (Pa)
e_KNOWN = 1.602176634e-19   # elementary charge  (C) — for comparison


# ═══════════════════════════════════════════════════════════════════════════════
#  10 kΩ NTC Thermistor  →  Temperature
# ═══════════════════════════════════════════════════════════════════════════════
#
# PASCO AP-8210A thermistor resistance table (Ω → °C) printed on apparatus.
# We store (resistance_Ω, temperature_°C) pairs in descending R order (R
# decreases as T increases for an NTC).
#
# If the user's values fall outside the table, we fall back to the
# Steinhart–Hart / B-parameter model:
#     1/T = 1/T₀ + (1/B)·ln(R/R₀)
# with  R₀ = 10 000 Ω  at  T₀ = 25 °C  (298.15 K), B = 3 950 K.

_THERM_TABLE = [
    # (R in Ω, T in °C)  — representative 10 kΩ NTC table for PASCO AP-8210A
    (19903, 10), (18963, 11), (18072, 12), (17226, 13), (16423, 14),
    (15659, 15), (14931, 16), (14237, 17), (13576, 18), (12945, 19),
    (12342, 20), (11766, 21), (11215, 22), (10688, 23), (10183, 24),
    (9700,  25), (9236,  26), (8792,  27), (8366,  28), (7957,  29),
    (7564,  30), (7186,  31), (6823,  32), (6473,  33), (6137,  34),
    (5813,  35), (5501,  36), (5201,  37), (4912,  38), (4633,  39),
    (4365,  40),
]

# Pre-compute for fast lookup
_THERM_R = np.array([row[0] for row in _THERM_TABLE], dtype=float)
_THERM_T = np.array([row[1] for row in _THERM_TABLE], dtype=float)

# B-parameter model constants
_NTC_R0 = 10_000.0   # Ω at T0
_NTC_T0 = 298.15      # K  (25 °C)
_NTC_B  = 3_950.0     # K


def thermistor_to_temp(R_ohm: float) -> float:
    """Convert 10 kΩ NTC thermistor resistance (Ω) → temperature (°C).

    Uses linear interpolation of the PASCO table when in range,
    otherwise falls back to the B-parameter model.
    """
    # Table range check (R is descending)
    if _THERM_R[-1] <= R_ohm <= _THERM_R[0]:
        # np.interp needs ascending x, so flip
        return float(np.interp(R_ohm, _THERM_R[::-1], _THERM_T[::-1]))
    # Fallback: B-parameter model
    T_K = 1.0 / (1.0 / _NTC_T0 + (1.0 / _NTC_B) * math.log(R_ohm / _NTC_R0))
    return T_K - 273.15


# ═══════════════════════════════════════════════════════════════════════════════
#  Air properties
# ═══════════════════════════════════════════════════════════════════════════════

# Air viscosity lookup (Pa·s) — PASCO manual Appendix A
_ETA_TABLE = {
    15: 1.802e-5, 16: 1.806e-5, 17: 1.810e-5, 18: 1.813e-5,
    19: 1.817e-5, 20: 1.821e-5, 21: 1.824e-5, 22: 1.828e-5,
    23: 1.832e-5, 24: 1.836e-5, 25: 1.840e-5, 26: 1.843e-5,
    27: 1.847e-5, 28: 1.851e-5, 29: 1.855e-5, 30: 1.859e-5,
}


def air_viscosity(T_celsius: float) -> float:
    """Dynamic viscosity of air (Pa·s) by interpolation of the PASCO table."""
    temps = sorted(_ETA_TABLE.keys())
    if T_celsius <= temps[0]:
        return _ETA_TABLE[temps[0]]
    if T_celsius >= temps[-1]:
        return _ETA_TABLE[temps[-1]]
    for i in range(len(temps) - 1):
        t0, t1 = temps[i], temps[i + 1]
        if t0 <= T_celsius <= t1:
            frac = (T_celsius - t0) / (t1 - t0)
            return _ETA_TABLE[t0] + frac * (_ETA_TABLE[t1] - _ETA_TABLE[t0])
    return _ETA_TABLE[20]


def air_density(T_celsius: float, P_Pa: float = P_ATM_DEFAULT) -> float:
    """Air density (kg/m³) from the ideal gas law."""
    M_air = 0.02897   # kg/mol
    R_gas = 8.314      # J/(mol·K)
    return P_Pa * M_air / (R_gas * (T_celsius + 273.15))


# ═══════════════════════════════════════════════════════════════════════════════
#  Core charge calculation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_charge(
    v_fall: float,       # m/s
    v_rise: float,       # m/s
    voltage: float,      # V
    d_sep: float,        # m
    eta: float,          # Pa·s
    rho_oil: float,      # kg/m³
    rho_air: float,      # kg/m³
    pressure: float,     # Pa
) -> dict:
    """Return dict with a1_m, q0_C, q_C, n_est, k_corr."""
    drho = rho_oil - rho_air

    # Uncorrected radius  (Eqn 5)
    a1 = math.sqrt(9.0 * eta * v_fall / (2.0 * drho * g))

    # Uncorrected charge  (Eqn 6)
    coeff = (18.0 * math.pi * d_sep) / voltage
    q0 = coeff * math.sqrt(eta**3 * v_fall / (2.0 * drho * g)) * (v_rise + v_fall)

    # Cunningham correction  (Eqn 10)
    corr = 1.0 + b_CUNNINGHAM / (pressure * a1)
    q = q0 / corr**1.5

    return {
        "a1_m":   a1,
        "q0_C":   q0,
        "q_C":    q,
        "n_est":  round(abs(q) / e_KNOWN),
        "k_corr": corr,
    }


def compute_charge_with_uncertainty(
    v_fall: float, v_rise: float,
    V_lo: float, V_hi: float,
    d_sep: float,
    T_lo: float, T_hi: float,        # °C from thermistor bounds
    rho_oil: float,
    pressure: float,
) -> dict:
    """
    Compute charge at nominal values and propagate uncertainty from
    voltage (V_lo, V_hi) and temperature (T_lo, T_hi) bounds.

    Uncertainty is estimated as  δq = (q_max − q_min) / 2  where q_max
    and q_min are the extreme charges over the four (V, T) corner
    combinations.
    """
    V_nom = (V_lo + V_hi) / 2.0
    T_nom = (T_lo + T_hi) / 2.0

    # Nominal calculation
    eta_nom = air_viscosity(T_nom)
    rho_air_nom = air_density(T_nom, pressure)
    nom = compute_charge(v_fall, v_rise, V_nom, d_sep,
                         eta_nom, rho_oil, rho_air_nom, pressure)

    # Evaluate at four corners of (V, T) parameter space
    corners = []
    for V in (V_lo, V_hi):
        for T in (T_lo, T_hi):
            eta_c = air_viscosity(T)
            rho_c = air_density(T, pressure)
            c = compute_charge(v_fall, v_rise, V, d_sep,
                               eta_c, rho_oil, rho_c, pressure)
            corners.append(c)

    q_vals = [abs(c["q_C"]) for c in corners]
    q_min, q_max = min(q_vals), max(q_vals)

    nom["delta_q_V_T"] = (q_max - q_min) / 2.0   # half-spread
    nom["q_min"]       = q_min
    nom["q_max"]       = q_max

    # Also break out individual contributions:
    # Temperature uncertainty (fix V at nominal)
    q_T = []
    for T in (T_lo, T_hi):
        eta_c = air_viscosity(T)
        rho_c = air_density(T, pressure)
        c = compute_charge(v_fall, v_rise, V_nom, d_sep,
                           eta_c, rho_oil, rho_c, pressure)
        q_T.append(abs(c["q_C"]))
    nom["delta_q_T"] = abs(q_T[1] - q_T[0]) / 2.0

    # Voltage uncertainty (fix T at nominal)
    q_V = []
    for V in (V_lo, V_hi):
        c = compute_charge(v_fall, v_rise, V, d_sep,
                           eta_nom, rho_oil, rho_air_nom, pressure)
        q_V.append(abs(c["q_C"]))
    nom["delta_q_V"] = abs(q_V[1] - q_V[0]) / 2.0

    # Combined in quadrature
    nom["delta_q_combined"] = math.sqrt(nom["delta_q_V"]**2 +
                                         nom["delta_q_T"]**2)
    return nom


# ═══════════════════════════════════════════════════════════════════════════════
#  Parse oscillation marks → rise / fall velocity pairs
# ═══════════════════════════════════════════════════════════════════════════════

def extract_velocities(marks: list[dict], reticle_dist: float,
                       fps: float) -> tuple[list[dict], list[dict], list[dict]]:
    """
    From alternating top/bottom marks, extract (v_fall, v_rise) pairs.

    Convention:
        top  → bottom  =  falling:  v_f = reticle_dist / Δt
        bottom → top   =  rising:   v_r = reticle_dist / Δt

    Returns (pairs, falls, rises).
    """
    if len(marks) < 2:
        return [], [], []

    intervals: list[dict] = []
    for i in range(len(marks) - 1):
        m0, m1 = marks[i], marks[i + 1]
        dt = abs(m1["time_s"] - m0["time_s"])
        if dt < 1e-6:
            continue
        intervals.append({
            "direction": f"{m0['type']}->{m1['type']}",
            "velocity":  reticle_dist / dt,
            "dt":        dt,
            "frame0":    m0["frame"],
            "frame1":    m1["frame"],
        })

    falls = [iv for iv in intervals if iv["direction"] == "top->bottom"]
    rises = [iv for iv in intervals if iv["direction"] == "bottom->top"]

    pairs = []
    if falls and rises:
        pairs.append({
            "v_fall": np.mean([f["velocity"] for f in falls]),
            "v_rise": np.mean([r["velocity"] for r in rises]),
            "t_fall": np.mean([f["dt"] for f in falls]),
            "t_rise": np.mean([r["dt"] for r in rises]),
            "n_fall": len(falls),
            "n_rise": len(rises),
            # Standard deviations on individual velocity measurements
            "v_fall_std": (np.std([f["velocity"] for f in falls], ddof=1)
                          if len(falls) > 1 else 0.0),
            "v_rise_std": (np.std([r["velocity"] for r in rises], ddof=1)
                          if len(rises) > 1 else 0.0),
        })

    return pairs, falls, rises


# ═══════════════════════════════════════════════════════════════════════════════
#  Main analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_file(json_path: str, params: dict) -> list[dict]:
    """Analyse all droplets in a marks JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    fps      = data["fps"]
    droplets = data.get("droplets", [])

    d_sep   = params["plate_separation_m"]
    d_ret   = params["reticle_distance_m"]
    P       = params["pressure_Pa"]
    rho_oil = params["rho_oil"]

    # Temperature — supports both direct °C and thermistor modes
    if "T_lo_C" in params:
        T_lo = params["T_lo_C"]
        T_hi = params["T_hi_C"]
    else:
        R_lo = params["thermistor_R_lo"]
        R_hi = params["thermistor_R_hi"]
        T_lo = thermistor_to_temp(R_hi)   # higher R → lower T for NTC
        T_hi = thermistor_to_temp(R_lo)   # lower R  → higher T
    T_nom = (T_lo + T_hi) / 2.0

    eta_nom     = air_viscosity(T_nom)
    rho_air_nom = air_density(T_nom, P)

    print(f"\n{'=' * 72}")
    print(f"  FILE: {os.path.basename(json_path)}")
    print(f"{'=' * 72}")
    print(f"  Plate separation  d     = {d_sep * 1e3:.3f} mm")
    print(f"  Reticle distance        = {d_ret * 1e3:.4f} mm")
    print(f"  Temperature             = {T_lo:.2f} - {T_hi:.2f} C  "
          f"(nom {T_nom:.2f} C)")
    print(f"  Pressure                = {P:.0f} Pa")
    print(f"  Air viscosity  eta(nom) = {eta_nom:.4e} Pa.s")
    print(f"  Air density   rho_air   = {rho_air_nom:.3f} kg/m3")
    print(f"  Oil density   rho_oil   = {rho_oil:.1f} kg/m3")
    print(f"{'=' * 72}\n")

    results = []

    for drop in droplets:
        drop_id = drop["id"]
        marks   = drop.get("marks", [])

        # -- Per-droplet voltage (low / high) -----------------------------
        V_lo = drop.get("voltage_lo_V")
        V_hi = drop.get("voltage_hi_V")
        V_old = drop.get("voltage_V")   # legacy single value

        if V_lo is None or V_hi is None:
            # If only legacy single value exists, use it +/- 0
            if V_old is not None and V_old != 0:
                V_lo = V_hi = float(V_old)
            # Use global voltage from params if available
            elif "voltage_lo_V" in params and "voltage_hi_V" in params:
                V_lo = params["voltage_lo_V"]
                V_hi = params["voltage_hi_V"]
            else:
                print(f"  Droplet #{drop_id}: voltage not set. "
                      f"Enter now or skip.")
                try:
                    lo = input(f"    V_low  for droplet #{drop_id} (V) "
                               f"[Enter to skip]: ").strip()
                    hi = input(f"    V_high for droplet #{drop_id} (V) "
                               f"[Enter to skip]: ").strip()
                    if lo and hi:
                        V_lo, V_hi = float(lo), float(hi)
                    else:
                        print(f"    -> Skipped.\n")
                        continue
                except (ValueError, EOFError):
                    print(f"    -> Skipped.\n")
                    continue

        if V_lo > V_hi:
            V_lo, V_hi = V_hi, V_lo

        V_nom = (V_lo + V_hi) / 2.0

        if len(marks) < 3:
            print(f"  Droplet #{drop_id}: Not enough marks "
                  f"({len(marks)}), need >= 3. Skipped.\n")
            continue

        pairs, falls, rises = extract_velocities(marks, d_ret, fps)
        if not pairs:
            print(f"  Droplet #{drop_id}: Could not extract "
                  f"velocity pairs. Skipped.\n")
            continue

        pair = pairs[0]

        # -- Charge with uncertainty --------------------------------------
        res = compute_charge_with_uncertainty(
            v_fall=pair["v_fall"], v_rise=pair["v_rise"],
            V_lo=V_lo, V_hi=V_hi,
            d_sep=d_sep,
            T_lo=T_lo, T_hi=T_hi,
            rho_oil=rho_oil, pressure=P,
        )

        # Velocity-based statistical uncertainty
        # dq/q ~ contributions from dv_f and dv_r  (first-order)
        nf, nr = pair["n_fall"], pair["n_rise"]
        se_vf = pair["v_fall_std"] / math.sqrt(nf) if nf > 1 else 0
        se_vr = pair["v_rise_std"] / math.sqrt(nr) if nr > 1 else 0
        vf, vr = pair["v_fall"], pair["v_rise"]
        # From Eqn 6,  q ~ v_f^(3/2) * (v_r + v_f)  approximately
        # So  dq/q ~ sqrt( (3/(2*v_f) * dv_f)^2 + (1/(v_r+v_f) * d(v_r+v_f))^2 )
        frac_vf = (1.5 * se_vf / vf) if vf > 0 else 0
        frac_vsum = (math.sqrt(se_vf**2 + se_vr**2) / (vf + vr)
                     if (vf + vr) > 0 else 0)
        delta_q_stat = abs(res["q_C"]) * math.sqrt(frac_vf**2 + frac_vsum**2)

        # Total uncertainty (stat + systematic in quadrature)
        delta_q_total = math.sqrt(delta_q_stat**2 +
                                   res["delta_q_combined"]**2)

        # Store everything
        res["droplet_id"]    = drop_id
        res["voltage_lo_V"]  = V_lo
        res["voltage_hi_V"]  = V_hi
        res["voltage_nom_V"] = V_nom
        res["T_lo_C"]        = T_lo
        res["T_hi_C"]        = T_hi
        res["T_nom_C"]       = T_nom
        res["v_fall"]        = vf
        res["v_rise"]        = vr
        res["t_fall_avg"]    = pair["t_fall"]
        res["t_rise_avg"]    = pair["t_rise"]
        res["n_fall_meas"]   = nf
        res["n_rise_meas"]   = nr
        res["delta_q_stat"]  = delta_q_stat
        res["delta_q_total"] = delta_q_total
        results.append(res)

        # -- Pretty-print -------------------------------------------------
        q = res["q_C"]
        dq = delta_q_total
        print(f"  +-- Droplet #{drop_id}  "
              f"(V = {V_lo:.1f}-{V_hi:.1f} V, nom {V_nom:.1f} V)")
        print(f"  |  Falls: {nf}   avg t_f = {pair['t_fall']:.3f} s   "
              f"v_f = {vf*1e4:.4f} x10^-4 m/s")
        print(f"  |  Rises: {nr}   avg t_r = {pair['t_rise']:.3f} s   "
              f"v_r = {vr*1e4:.4f} x10^-4 m/s")
        print(f"  |  Radius a1            = {res['a1_m']*1e6:.3f} um")
        print(f"  |  Cunningham factor     = {res['k_corr']:.4f}")
        print(f"  |  q (corrected)        = {q:.4e} C")
        print(f"  |  dq (voltage)         = {res['delta_q_V']:.2e} C")
        print(f"  |  dq (temperature)     = {res['delta_q_T']:.2e} C")
        print(f"  |  dq (systematic V+T)  = {res['delta_q_combined']:.2e} C")
        print(f"  |  dq (statistical)     = {delta_q_stat:.2e} C")
        print(f"  |  dq (total)           = {dq:.2e} C")
        print(f"  |  q +/- dq             = ({q:.4e} +/- {dq:.2e}) C")
        print(f"  |  n_est = {res['n_est']}   "
              f"(q/e = {abs(q) / e_KNOWN:.3f} +/- "
              f"{dq / e_KNOWN:.3f})")
        print(f"  +{'=' * 57}\n")

        # Individual intervals
        if len(falls) > 1 or len(rises) > 1:
            print(f"      Fall velocities (x10^-4 m/s):")
            for i, f_iv in enumerate(falls, 1):
                print(f"        #{i}: dt={f_iv['dt']:.3f}s  "
                      f"v={f_iv['velocity']*1e4:.4f}")
            print(f"      Rise velocities (x10^-4 m/s):")
            for i, r_iv in enumerate(rises, 1):
                print(f"        #{i}: dt={r_iv['dt']:.3f}s  "
                      f"v={r_iv['velocity']*1e4:.4f}")
            print()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def make_histogram(all_results: list[dict], output_dir: str):
    """Histogram of charges + q/e scatter with error bars."""
    if not all_results:
        print("No results to plot.")
        return

    charges   = [abs(r["q_C"]) for r in all_results]
    charges_e = [abs(r["q_C"]) / e_KNOWN for r in all_results]
    errors_e  = [r["delta_q_total"] / e_KNOWN for r in all_results]
    n_values  = [r["n_est"] for r in all_results]
    max_n     = max(n_values) if n_values else 5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Histogram --------------------------------------------------------
    ax1 = axes[0]
    charges_19 = [q * 1e19 for q in charges]
    bins = max(5, int(len(charges_19) ** 0.5) + 1)
    ax1.hist(charges_19, bins=bins, edgecolor="black", alpha=0.7,
             color="steelblue")
    for n in range(1, max_n + 2):
        ls = "--" if n == 1 else ":"
        lw = 1.5 if n == 1 else 0.8
        al = 1.0 if n == 1 else 0.5
        lbl = f"e = {e_KNOWN*1e19:.3f}" if n == 1 else None
        ax1.axvline(n * e_KNOWN * 1e19, color="red", linestyle=ls,
                    linewidth=lw, alpha=al, label=lbl)
    ax1.set_xlabel("|q|  (x10^-19 C)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Measured Charges")
    ax1.legend()

    # -- q / e scatter with error bars ------------------------------------
    ax2 = axes[1]
    drop_ids = [r["droplet_id"] for r in all_results]
    ax2.errorbar(drop_ids, charges_e, yerr=errors_e, fmt="o",
                 color="steelblue", ecolor="gray", elinewidth=1,
                 capsize=3, markersize=5, markeredgecolor="black", zorder=3)
    for n in range(1, max_n + 2):
        ax2.axhline(n, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("Droplet #")
    ax2.set_ylabel("|q| / e")
    ax2.set_title("Charge Quantisation  (q / e)  with uncertainties")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "charge_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[PLOT SAVED] {plot_path}")


def make_summary_table(all_results: list[dict], output_dir: str):
    """Save a summary CSV."""
    csv_path = os.path.join(output_dir, "charge_results.csv")
    fields = [
        "droplet_id", "voltage_lo_V", "voltage_hi_V", "voltage_nom_V",
        "T_lo_C", "T_hi_C", "T_nom_C",
        "t_fall_avg", "t_rise_avg", "n_fall_meas", "n_rise_meas",
        "v_fall", "v_rise",
        "a1_m", "q0_C", "q_C", "n_est", "q_over_e",
        "delta_q_V", "delta_q_T", "delta_q_combined",
        "delta_q_stat", "delta_q_total",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            row = dict(r)
            row["q_over_e"] = abs(r["q_C"]) / e_KNOWN
            writer.writerow(row)
    print(f"[CSV SAVED] {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive parameter entry
# ═══════════════════════════════════════════════════════════════════════════════

def get_params_interactive() -> dict:
    """Prompt the user for experimental parameters."""
    print("\n" + "=" * 62)
    print("  Experimental Parameters")
    print("=" * 62)

    def ask(prompt, default, cast=float):
        try:
            val = input(f"  {prompt} [{default}]: ").strip()
            return cast(val) if val else default
        except (ValueError, EOFError):
            return default

    d_sep_mm = ask("Plate separation d (mm)", 7.59)
    d_ret_mm = ask("Reticle distance per major div (mm)", 0.500)
    n_div    = ask("Number of major divisions traversed", 1.0)

    print()
    print("  -- Temperature --")
    print("  Choose input mode:")
    print("    1) Direct temperature (°C)")
    print("    2) 10 kΩ NTC thermistor resistance (Ω)")
    mode = input("  Mode [1]: ").strip()

    if mode == "2":
        R_lo = ask("Thermistor resistance LOW  (Ohm)", 11215)
        R_hi = ask("Thermistor resistance HIGH (Ohm)", 11215)
        if R_lo > R_hi:
            R_lo, R_hi = R_hi, R_lo
        T_lo = thermistor_to_temp(R_hi)   # higher R -> lower T
        T_hi = thermistor_to_temp(R_lo)
    else:
        T_lo = ask("Temperature LOW  (°C)", 22.2)
        T_hi = ask("Temperature HIGH (°C)", 22.5)
        if T_lo > T_hi:
            T_lo, T_hi = T_hi, T_lo

    T_nom = (T_lo + T_hi) / 2.0
    print(f"    -> Temperature range: {T_lo:.2f} - {T_hi:.2f} °C  "
          f"(nom {T_nom:.2f} °C)")

    print()
    print("  -- Pressure --")
    print("  Choose input unit:")
    print("    1) cmHg")
    print("    2) Pa")
    p_mode = input("  Unit [1]: ").strip()
    if p_mode == "2":
        P = ask("Atmospheric pressure (Pa)", P_ATM_DEFAULT)
    else:
        P_cmHg = ask("Atmospheric pressure (cmHg)", 75.9)
        P = P_cmHg * 1333.22        # 1 cmHg = 1333.22 Pa
        print(f"    -> {P_cmHg:.2f} cmHg = {P:.0f} Pa")

    rho = ask("Oil density (kg/m3)", RHO_OIL)

    print()
    print("  -- Global voltage (applied to droplets without per-droplet V) --")
    V_lo = ask("Voltage LOW  (V)", 552.0)
    V_hi = ask("Voltage HIGH (V)", 555.0)
    if V_lo > V_hi:
        V_lo, V_hi = V_hi, V_lo

    return {
        "plate_separation_m": d_sep_mm * 1e-3,
        "reticle_distance_m": d_ret_mm * n_div * 1e-3,
        "T_lo_C":             T_lo,
        "T_hi_C":             T_hi,
        "pressure_Pa":        P,
        "rho_oil":            rho,
        "voltage_lo_V":       V_lo,
        "voltage_hi_V":       V_hi,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # -- Collect JSON files -----------------------------------------------
    if len(sys.argv) > 1:
        json_files = sys.argv[1:]
    else:
        data_dir = Path(__file__).parent / "data"
        json_files = sorted(data_dir.glob("*_marks.json"))
        if not json_files:
            print("No *_marks.json files found in data/. "
                  "Run video_player.py first.")
            sys.exit(1)
        json_files = [str(f) for f in json_files]
        print(f"\nFound {len(json_files)} marks file(s):")
        for f in json_files:
            print(f"  - {os.path.basename(f)}")

    # -- Get parameters ---------------------------------------------------
    params = get_params_interactive()

    # -- Analyse each file ------------------------------------------------
    all_results = []
    for jf in json_files:
        results = analyse_file(str(jf), params)
        all_results.extend(results)

    if not all_results:
        print("\nNo valid droplet data found. Mark at least 3 "
              "alternating T/B positions per droplet.")
        sys.exit(0)

    # -- Summary ----------------------------------------------------------
    charges = [abs(r["q_C"]) for r in all_results]
    errors  = [r["delta_q_total"] for r in all_results]
    n_vals  = [r["n_est"] for r in all_results]

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Droplets analysed: {len(all_results)}")
    print(f"  Charge range:  {min(charges):.4e}  ->  {max(charges):.4e} C")
    print(f"  Estimated n:   {min(n_vals)}  ->  {max(n_vals)}")

    # Group by n
    unique_n = sorted(set(n_vals))
    hdr = (f"  {'n':>3}  {'Cnt':>3}  {'Mean q':>14}  {'sigma_q':>12}  "
           f"{'<dq_tot>':>12}  {'q/n':>14}  {'q/n / e':>8}")
    print(f"\n{hdr}")
    print(f"  {'---':>3}  {'---':>3}  {'---':>14}  {'---':>12}  "
          f"{'---':>12}  {'---':>14}  {'---':>8}")

    e_ests = []
    e_errs = []
    for n in unique_n:
        idx = [i for i, r in enumerate(all_results) if r["n_est"] == n]
        grp_q  = [charges[i] for i in idx]
        grp_dq = [errors[i]  for i in idx]
        mean_q = np.mean(grp_q)
        std_q  = np.std(grp_q, ddof=1) if len(grp_q) > 1 else 0
        avg_dq = np.mean(grp_dq)
        e_est  = mean_q / n if n > 0 else 0
        e_rat  = e_est / e_KNOWN if n > 0 else 0
        if n > 0:
            e_ests.append(e_est)
            e_err_n = math.sqrt(std_q**2 + avg_dq**2) / n
            e_errs.append(e_err_n)
        print(f"  {n:3d}  {len(grp_q):3d}  {mean_q:14.4e}  "
              f"{std_q:12.4e}  {avg_dq:12.4e}  "
              f"{e_est:14.4e}  {e_rat:8.4f}")

    if e_ests:
        # Weighted average  (weight = 1/sigma^2)
        w = np.array([1.0 / (s**2) if s > 0 else 1e30 for s in e_errs])
        e_avg = float(np.average(e_ests, weights=w))
        e_err = 1.0 / math.sqrt(float(np.sum(w)))
        pct = abs(e_avg - e_KNOWN) / e_KNOWN * 100

        print(f"\n  ======================================================")
        print(f"  Estimated e  = ({e_avg:.4e} +/- {e_err:.2e}) C")
        print(f"  Known     e  =  {e_KNOWN:.6e} C")
        print(f"  Percent error = {pct:.2f}%")
        print(f"  ======================================================\n")

    # -- Output files -----------------------------------------------------
    output_dir = str(Path(json_files[0]).parent)
    make_summary_table(all_results, output_dir)
    make_histogram(all_results, output_dir)


if __name__ == "__main__":
    main()
