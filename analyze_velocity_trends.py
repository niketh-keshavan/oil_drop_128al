"""Analyze individual cycle velocities to detect mid-measurement charge changes."""
import json
import numpy as np

RETICLE = 0.5e-3  # m

for trial in [1, 2, 3]:
    with open(f'data/TRIAL_{trial}_marks.json') as f:
        data = json.load(f)
    fps = data['fps']
    print(f'=== TRIAL {trial} (fps={fps:.4f}) ===')
    for drop in data['droplets']:
        marks = drop['marks']
        print(f'  Drop {drop["id"]}:')
        falls, rises = [], []
        for i in range(len(marks)-1):
            m0, m1 = marks[i], marks[i+1]
            dt = abs(m1['time_s'] - m0['time_s'])
            v = RETICLE / dt * 1e3  # mm/s
            direction = f"{m0['type']}->{m1['type']}"
            t_mid = (m0['time_s'] + m1['time_s']) / 2.0
            if direction == 'top->bottom':
                falls.append((i, t_mid, v, dt))
                tag = 'FALL'
            elif direction == 'bottom->top':
                rises.append((i, t_mid, v, dt))
                tag = 'RISE'
            else:
                tag = '????'
            print(f'    {tag} #{i}: t_mid={t_mid:6.1f}s  v={v:.4f} mm/s  dt={dt:.3f}s')

        if falls:
            vf = [x[2] for x in falls]
            print(f'    FALL summary: mean={np.mean(vf):.4f} std={np.std(vf,ddof=1):.4f} '
                  f'cv={np.std(vf,ddof=1)/np.mean(vf)*100:.1f}%')
        if rises:
            vr = [x[2] for x in rises]
            print(f'    RISE summary: mean={np.mean(vr):.4f} std={np.std(vr,ddof=1):.4f} '
                  f'cv={np.std(vr,ddof=1)/np.mean(vr)*100:.1f}%')

        # Check for systematic trend in rise velocities
        if len(rises) > 2:
            vr_arr = np.array([x[2] for x in rises])
            t_arr = np.array([x[1] for x in rises])
            coeffs = np.polyfit(t_arr, vr_arr, 1)
            print(f'    RISE trend: slope={coeffs[0]*1000:.3f} (mm/s)/min')

            # Check for step changes: compare consecutive rise velocities
            print(f'    RISE step changes:')
            for j in range(1, len(vr_arr)):
                delta = vr_arr[j] - vr_arr[j-1]
                pct = delta / vr_arr[j-1] * 100
                flag = ' *** POSSIBLE CHARGE CHANGE' if abs(pct) > 20 else ''
                print(f'      cycle {j-1}->{j}: Δv = {delta:+.4f} mm/s ({pct:+.1f}%){flag}')

        # Same for fall
        if len(falls) > 2:
            vf_arr = np.array([x[2] for x in falls])
            print(f'    FALL step changes:')
            for j in range(1, len(vf_arr)):
                delta = vf_arr[j] - vf_arr[j-1]
                pct = delta / vf_arr[j-1] * 100
                flag = ' *** UNEXPECTED' if abs(pct) > 20 else ''
                print(f'      cycle {j-1}->{j}: Δv = {delta:+.4f} mm/s ({pct:+.1f}%){flag}')
        print()
