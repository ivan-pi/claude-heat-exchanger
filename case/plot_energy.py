#!/usr/bin/env python3
"""Post-processing for the coupled heat-exchanger case.

Reads the outputs of both participants and produces
  * energy_balance.png   temperature field, interface profiles (both meshes),
                         heat rates over time, balance residuals
  * energy_summary.txt   final-time energy budget

Energy budget (per unit depth, W/m), with H = enthalpy flow relative to T_ref:
  hot channel :  dE_h/dt = (H_in - H_out)_h + Q_wall,h          (Q_wall,h < 0)
  cold channel:  dE_c/dt = (H_in - H_out)_c + Q_wall,c
  solid wall  :  dE_s/dt = -(Q_hot + Q_cold)                    (Q: wall -> fluid)
  interfaces  :  Q_wall (fluid mesh, after mapping) == Q (solid mesh)
  steady state:  hot-stream loss == heat through wall == cold-stream gain

Usage: python3 plot_energy.py [case-dir]
"""
import os
import sys

import numpy as np

from hx_post import (AQUA, BLUE, INK, INK2, MAGENTA, MM, ORANGE, SURF, VIOLET, YELLOW, Field,
                     apply_style, case_dirs, draw_fields, load_energy_logs, plt, read_csv,
                     sorted_files, temperature_cmap)

CASE, FL, SO = case_dirs(sys.argv)
apply_style()

fe, se = load_energy_logs(FL, SO)
fi = read_csv(os.path.join(FL, "fluid_interface.csv"))
si = read_csv(os.path.join(SO, "solid_interface.csv"))
t = fe["time"]

# --- budgets --------------------------------------------------------------------------
hot_loss = fe["hot_H_net_in"]            # enthalpy given up by the hot stream
cold_gain = -fe["cold_H_net_in"]        # enthalpy picked up by the cold stream
wall_in_s = -se["Q_into_hot_fluid"]     # hot fluid -> wall   (solid mesh)
wall_out_s = se["Q_into_cold_fluid"]    # wall -> cold fluid  (solid mesh)
wall_in_f = -fe["hot_Q_wall"]           # same, integrated on the fluid mesh after mapping
wall_out_f = fe["cold_Q_wall"]

scale = np.maximum(np.abs(wall_out_s), 1e-30)
res_hot = fe["hot_dEdt"] - fe["hot_H_net_in"] - fe["hot_Q_wall"]
res_cold = fe["cold_dEdt"] - fe["cold_H_net_in"] - fe["cold_Q_wall"]
res_solid = se["dEdt_solid"] + se["Q_into_hot_fluid"] + se["Q_into_cold_fluid"]
map_hot = wall_in_f - wall_in_s     # coupling residual (+ mapping non-conservation)
map_cold = wall_out_f - wall_out_s
# whole system: enthalpy brought in by both streams = storage in both channels + wall
res_global = (fe["hot_H_net_in"] + fe["cold_H_net_in"]
              - fe["hot_dEdt"] - fe["cold_dEdt"] - se["dEdt_solid"])

# --- figure ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12.5, 13.0), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0])

# (a) temperature field: diverging around the mean inlet temperature
ax = fig.add_subplot(gs[0, :])
hot = Field(sorted_files(os.path.join(FL, "fluid_hot_*.vtk"))[-1])
cold = Field(sorted_files(os.path.join(FL, "fluid_cold_*.vtk"))[-1])
solid = Field(sorted_files(os.path.join(SO, "solid_*.vtk"))[-1])
Tmin = min(np.nanmin(f.T) for f in (hot, cold, solid))
Tmax = max(np.nanmax(f.T) for f in (hot, cold, solid))
Tmid = 0.5 * (Tmin + Tmax)
half = max(Tmax - Tmid, Tmid - Tmin)
im = draw_fields(ax, (hot, cold, solid), temperature_cmap(), Tmid - half, Tmid + half,
                 x_max=hot.extent[1] + hot.dx / 2)
ax.set_ylim(0, (hot.extent[3] + hot.dx / 2) * MM)
ax.set_title(f"(a) T and streamlines, t = {t[-1]:.1f} s:  hot channel (→, top), insulating wall, "
             "cold channel (←, bottom); baffles dark grey", loc="left", color=INK)
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
cb.set_label("T [K]")

# (b) interface heat flux, both meshes
ax = fig.add_subplot(gs[1, 0])
ax.plot(fi["x"] * MM, -fi["q_hot"] / 1e3, color=ORANGE, label="hot fluid → wall  (Fluid mesh)")
ax.plot(si["x"] * MM, -si["q_hot"] / 1e3, ls="none", marker="o", ms=4, mfc=SURF, mec=ORANGE, mew=1.2,
        label="hot fluid → wall  (Solid mesh)")
ax.plot(fi["x"] * MM, fi["q_cold"] / 1e3, color=BLUE, label="wall → cold fluid  (Fluid mesh)")
ax.plot(si["x"] * MM, si["q_cold"] / 1e3, ls="none", marker="s", ms=4, mfc=SURF, mec=BLUE, mew=1.2,
        label="wall → cold fluid  (Solid mesh)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("heat flux [kW/m²]")
ax.set_title("(b) Interface heat flux at the final time", loc="left", color=INK)
ax.legend(fontsize=8.5, loc="best")

# (c) interface temperature, both meshes
ax = fig.add_subplot(gs[1, 1])
ax.plot(fi["x"] * MM, fi["Tw_hot"], color=ORANGE, label="hot interface  (Fluid writes)")
ax.plot(si["x"] * MM, si["T_hot"], ls="none", marker="o", ms=4, mfc=SURF, mec=ORANGE, mew=1.2,
        label="hot interface  (Solid reads, mapped)")
ax.plot(fi["x"] * MM, fi["Tw_cold"], color=BLUE, label="cold interface  (Fluid writes)")
ax.plot(si["x"] * MM, si["T_cold"], ls="none", marker="s", ms=4, mfc=SURF, mec=BLUE, mew=1.2,
        label="cold interface  (Solid reads, mapped)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("T [K]")
ax.set_title("(c) Interface temperature at the final time", loc="left", color=INK)
ax.legend(fontsize=8.5, loc="best")

# (d) heat rates over time
ax = fig.add_subplot(gs[2, 0])
ax.plot(t, hot_loss, color=ORANGE, label="hot stream: H_in − H_out")
ax.plot(t, wall_in_s, color=YELLOW, label="hot fluid → wall")
ax.plot(t, wall_out_s, color=AQUA, label="wall → cold fluid")
ax.plot(t, cold_gain, color=BLUE, label="cold stream: H_out − H_in")
ax.set_xlabel("t [s]")
ax.set_ylabel("heat rate per unit depth [W/m]")
ax.set_title("(d) Heat rates: equal once the system is steady", loc="left", color=INK)
ax.set_ylim(min(0.0, 1.1 * np.min([hot_loss, cold_gain])), 1.6 * abs(wall_out_s[-1]))
ax.legend(fontsize=8.5, loc="lower right")

# (e) residuals relative to the heat through the wall
ax = fig.add_subplot(gs[2, 1])
eps = 1e-16
ax.semilogy(t, np.abs(res_hot) / scale + eps, color=ORANGE, label="hot channel: dE/dt − ΔH − Q_wall")
ax.semilogy(t, np.abs(res_cold) / scale + eps, color=BLUE, label="cold channel: dE/dt − ΔH − Q_wall")
ax.semilogy(t, np.abs(res_solid) / scale + eps, color=AQUA, label="solid: dE/dt + Q_hot + Q_cold")
ax.semilogy(t, np.abs(map_hot) / scale + eps, color=VIOLET, ls="--",
            label="hot interface: Q(Fluid mesh) − Q(Solid mesh)")
ax.semilogy(t, np.abs(map_cold) / scale + eps, color=MAGENTA, ls="--",
            label="cold interface: Q(Fluid mesh) − Q(Solid mesh)")
ax.semilogy(t, np.abs(res_global) / scale + eps, color=INK2, lw=1.5,
            label="global: ΣH_net,in − Σ dE/dt (both channels + wall)")
ax.set_xlabel("t [s]")
ax.set_ylabel("|residual| / Q_wall→cold")
ax.set_title("(e) Energy-balance residuals (relative)", loc="left", color=INK)
ax.set_ylim(1e-17, 1e1)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

out = os.path.join(CASE, "energy_balance.png")
fig.savefig(out, dpi=130)

# --- summary --------------------------------------------------------------------------
k = -1
rows = [
    ("hot stream loss      H_in - H_out", hot_loss[k]),
    ("hot fluid -> wall    (Fluid mesh)", wall_in_f[k]),
    ("hot fluid -> wall    (Solid mesh)", wall_in_s[k]),
    ("wall -> cold fluid   (Solid mesh)", wall_out_s[k]),
    ("wall -> cold fluid   (Fluid mesh)", wall_out_f[k]),
    ("cold stream gain     H_out - H_in", cold_gain[k]),
    ("solid storage        dE_s/dt", se["dEdt_solid"][k]),
    ("hot channel storage  dE_h/dt", fe["hot_dEdt"][k]),
    ("cold channel storage dE_c/dt", fe["cold_dEdt"][k]),
]
lines = [f"Energy budget at t = {t[k]:.3f} s  (W per m depth)"]
lines += [f"  {name:36s} {val:12.5f}" for name, val in rows]
lines.append(f"  hot loss - cold gain (steady-state check) / Q     = {(hot_loss[k] - cold_gain[k]) / cold_gain[k]:+.3e}")
lines.append(f"  max |global residual| over the run / Q            = {np.max(np.abs(res_global)) / abs(wall_out_s[k]):.3e}")
lines.append(f"  max |hot, cold, solid residual| over the run / Q  = {np.max(np.abs(res_hot)) / abs(wall_out_s[k]):.3e}, "
             f"{np.max(np.abs(res_cold)) / abs(wall_out_s[k]):.3e}, {np.max(np.abs(res_solid)) / abs(wall_out_s[k]):.3e}")
lines.append(f"  interface mismatch hot  (Fluid - Solid) / Q       = {map_hot[k] / wall_in_s[k]:+.3e}")
lines.append(f"  interface mismatch cold (Fluid - Solid) / Q       = {map_cold[k] / wall_out_s[k]:+.3e}")
lines.append(f"  bulk outlet temperatures: hot {fe['hot_Tb_out'][k]:.3f} K, cold {fe['cold_Tb_out'][k]:.3f} K")
text = "\n".join(lines)
with open(os.path.join(CASE, "energy_summary.txt"), "w") as f:
    f.write(text + "\n")
print(text)
print(f"wrote {out}")
