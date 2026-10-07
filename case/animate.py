#!/usr/bin/env python3
"""Animation of the coupled heat-exchanger run (MP4 via ffmpeg).

Top:          temperature field of both channels and the wall, with streamlines.
Bottom left:  heat rates over time (hot-stream loss, heat through the wall, cold-stream gain).
Bottom right: bulk (mixing-cup, flow-weighted) temperature along x for both streams and
              the wall mid-plane temperature.

Needs the VTK snapshots of both participants at matching times; set e.g.
`output_interval 0.1` in params.txt for a smooth animation.

Usage: python3 animate.py [case-dir] [output.mp4] [fps]
"""
import os
import sys

import numpy as np
from matplotlib.animation import FFMpegWriter

from hx_post import (AQUA, BLUE, INK, INK2, MM, ORANGE, Field, apply_style, case_dirs, draw_fields,
                     load_energy_logs, plt, sorted_files, temperature_cmap)

CASE, FL, SO = case_dirs(sys.argv)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CASE, "heat_exchanger.mp4")
FPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
apply_style()

hot_files = sorted_files(os.path.join(FL, "fluid_hot_*.vtk"))
cold_files = sorted_files(os.path.join(FL, "fluid_cold_*.vtk"))
solid_files = sorted_files(os.path.join(SO, "solid_*.vtk"))
nframes = min(len(hot_files), len(cold_files), len(solid_files))

fe, se = load_energy_logs(FL, SO)
t_log = fe["time"]
hot_loss = fe["hot_H_net_in"]
cold_gain = -fe["cold_H_net_in"]
wall_q = se["Q_into_cold_fluid"]

# snapshot times: t = 0 and then every output interval (matches the logs)
out_dt = t_log[-1] / (nframes - 1) if nframes > 1 else 0.0
t_frames = np.arange(nframes) * out_dt

T_LO, T_HI = 290.0, 350.0
cmap = temperature_cmap()

fig = plt.figure(figsize=(12.8, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.38, wspace=0.18,
                      left=0.06, right=0.93, top=0.92, bottom=0.08)
ax_f = fig.add_subplot(gs[0, :])
ax_q = fig.add_subplot(gs[1, 0])
ax_x = fig.add_subplot(gs[1, 1])
cax = fig.add_axes([0.945, 0.56, 0.012, 0.34])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(T_LO, T_HI))
cb = fig.colorbar(sm, cax=cax)
cb.set_label("T [K]")
title = fig.suptitle("", x=0.06, ha="left", fontsize=13, color=INK)

# heat-rate history (static curves + moving cursor)
ax_q.plot(t_log, hot_loss, color=ORANGE, label="hot stream: H_in − H_out")
ax_q.plot(t_log, wall_q, color=AQUA, label="through the wall")
ax_q.plot(t_log, cold_gain, color=BLUE, label="cold stream: H_out − H_in")
ax_q.set_xlim(0, t_log[-1])
ax_q.set_ylim(0, 1.6 * abs(wall_q[-1]))
ax_q.set_xlabel("t [s]")
ax_q.set_ylabel("heat rate [W per m depth]")
ax_q.set_title("Heat rates", loc="left", color=INK)
ax_q.legend(fontsize=8.5, loc="upper right")
cursor = ax_q.axvline(0.0, color=INK2, lw=1.0)

# profiles along x
x_lines = {
    "hot": ax_x.plot([], [], color=ORANGE, label="hot stream (bulk)")[0],
    "wall": ax_x.plot([], [], color=INK2, label="wall (mid-plane)")[0],
    "cold": ax_x.plot([], [], color=BLUE, label="cold stream (bulk)")[0],
}
ax_x.set_ylim(T_LO - 2, T_HI + 2)
ax_x.set_xlabel("x [mm]")
ax_x.set_ylabel("T [K]")
ax_x.set_title("Temperature along the exchanger", loc="left", color=INK, pad=22)
ax_x.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2)


def draw_frame(k):
    hot, cold, solid = Field(hot_files[k]), Field(cold_files[k]), Field(solid_files[k])
    L = hot.extent[1] - 0.5 * hot.dx  # node centres run from -dx/2 to L + dx/2

    ax_f.clear()
    draw_fields(ax_f, (hot, cold, solid), cmap, T_LO, T_HI, x_max=L)
    ax_f.set_ylim(0, (hot.extent[3] + hot.dx / 2) * MM)
    ax_f.text(0.995, 1.02, "hot channel →   |   insulating wall   |   ← cold channel     (baffles dark grey)",
              transform=ax_f.transAxes, ha="right", va="bottom", fontsize=9, color=INK2)

    tk = t_frames[k]
    title.set_text(f"Baffled counterflow heat exchanger: LBM fluid ⇄ preCICE ⇄ FD wall      t = {tk:5.1f} s")
    cursor.set_xdata([tk, tk])

    xs_f = hot.x_nodes * MM
    sel = (xs_f >= 0) & (xs_f <= L * MM)
    for key, fld in (("hot", hot), ("cold", cold)):
        x_lines[key].set_data(xs_f[sel], fld.bulk_temperature()[sel])
    x_lines["wall"].set_data(solid.x_nodes * MM, solid.T[solid.T.shape[0] // 2])
    ax_x.set_xlim(0, L * MM)


writer = FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with writer.saving(fig, OUT, dpi=110):
    for k in range(nframes):
        draw_frame(k)
        writer.grab_frame()
print(f"wrote {OUT} ({nframes} frames, {nframes / FPS:.1f} s at {FPS} fps)")
