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
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

CASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CASE, "heat_exchanger.mp4")
FPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
FL = os.path.join(CASE, "fluid", "output")
SO = os.path.join(CASE, "solid", "output")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "font.size": 10, "legend.frameon": False,
})


def read_vtk(path):
    with open(path) as f:
        lines = f.read().split("\n")
    nx, ny = (int(v) for v in lines[4].split()[1:3])
    ox, oy = (float(v) for v in lines[5].split()[1:3])
    dx, dy = (float(v) for v in lines[6].split()[1:3])
    n = nx * ny
    T, vel = None, None
    for k, line in enumerate(lines):
        if line.startswith("SCALARS T "):
            T = np.array(lines[k + 2:k + 2 + n], dtype=float).reshape(ny, nx)
        elif line.startswith("VECTORS velocity"):
            vel = np.array([v.split()[:2] for v in lines[k + 1:k + 1 + n]], dtype=float).reshape(ny, nx, 2)
    return T, vel, (ox, ox + (nx - 1) * dx, oy, oy + (ny - 1) * dy), dx, dy


hot_files = sorted(glob.glob(os.path.join(FL, "fluid_hot_*.vtk")))
cold_files = sorted(glob.glob(os.path.join(FL, "fluid_cold_*.vtk")))
solid_files = sorted(glob.glob(os.path.join(SO, "solid_*.vtk")))
nframes = min(len(hot_files), len(cold_files), len(solid_files))
if nframes == 0:
    sys.exit("no VTK snapshots found")

fe = np.genfromtxt(os.path.join(FL, "fluid_energy.csv"), delimiter=",", names=True)
se = np.genfromtxt(os.path.join(SO, "solid_energy.csv"), delimiter=",", names=True)
n = min(len(fe), len(se))
fe, se = fe[:n], se[:n]
t_log = fe["time"]
hot_loss = fe["hot_H_net_in"]
cold_gain = -fe["cold_H_net_in"]
wall_q = se["Q_into_cold_fluid"]

# snapshot times: t = 0 and then every output interval (matches the logs)
out_dt = t_log[-1] / (nframes - 1) if nframes > 1 else 0.0
t_frames = np.arange(nframes) * out_dt

T_LO, T_HI = 290.0, 350.0
cmap = LinearSegmentedColormap.from_list("cold-hot", [BLUE, "#e9e8e4", ORANGE])
cmap.set_bad("#3d3d3a")
mm = 1e3

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
    Th, Vh, eh, dx, _ = read_vtk(hot_files[k])
    Tc, Vc, ec, _, _ = read_vtk(cold_files[k])
    Ts, _, es, dxs, dys = read_vtk(solid_files[k])
    L = eh[1] + 0.5 * dx - dx  # node centres run from -dx/2 to L + dx/2

    ax_f.clear()
    ax_f.grid(False)
    for T, e, hx, hy in ((Th, eh, dx, dx), (Tc, ec, dx, dx), (Ts, es, dxs, dys)):
        ext = [(e[0] - hx / 2) * mm, (e[1] + hx / 2) * mm, (e[2] - hy / 2) * mm, (e[3] + hy / 2) * mm]
        ax_f.imshow(T, origin="lower", extent=ext, cmap=cmap, vmin=T_LO, vmax=T_HI,
                    aspect="auto", interpolation="nearest")
    for V, e in ((Vh, eh), (Vc, ec)):
        xs = np.linspace(e[0], e[1], V.shape[1]) * mm
        ys = np.linspace(e[2], e[3], V.shape[0]) * mm
        keep = (xs >= 0) & (xs <= L * mm)
        ax_f.streamplot(xs[keep], ys, V[:, keep, 0], V[:, keep, 1], density=(2.6, 1.1), color=INK,
                        linewidth=0.5, arrowsize=0.6)
    for yi in (es[2], es[3]):
        ax_f.axhline(yi * mm, color=INK, lw=0.8, ls="--")
    ax_f.set_xlim(0, L * mm)
    ax_f.set_ylim(0, (eh[3] + dx / 2) * mm)
    ax_f.set_aspect("equal")
    ax_f.set_xlabel("x [mm]")
    ax_f.set_ylabel("y [mm]")
    ax_f.text(0.995, 1.02, "hot channel →   |   insulating wall   |   ← cold channel     (baffles dark grey)",
              transform=ax_f.transAxes, ha="right", va="bottom", fontsize=9, color=INK2)

    tk = t_frames[k]
    title.set_text(f"Baffled counterflow heat exchanger: LBM fluid ⇄ preCICE ⇄ FD wall      t = {tk:5.1f} s")
    cursor.set_xdata([tk, tk])

    xs_f = (np.arange(Th.shape[1]) - 0.5) * dx * mm
    sel = (xs_f >= 0) & (xs_f <= L * mm)
    # mixing-cup temperature: sum(u T)/sum(u) per column; the column flow rate is the
    # same everywhere (mass conservation), so this is defined even with recirculation
    for key, T, V in (("hot", Th, Vh), ("cold", Tc, Vc)):
        u = V[:, :, 0]
        Tb = np.nansum(u * np.nan_to_num(T), axis=0) / np.sum(u, axis=0)
        x_lines[key].set_data(xs_f[sel], Tb[sel])
    xs_s = np.arange(Ts.shape[1]) * dxs * mm
    x_lines["wall"].set_data(xs_s, Ts[Ts.shape[0] // 2])
    ax_x.set_xlim(0, L * mm)


writer = FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with writer.saving(fig, OUT, dpi=110):
    for k in range(nframes):
        draw_frame(k)
        writer.grab_frame()
print(f"wrote {OUT} ({nframes} frames, {nframes / FPS:.1f} s at {FPS} fps)")
