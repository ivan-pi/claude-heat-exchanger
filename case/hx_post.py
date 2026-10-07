"""Shared post-processing helpers for the coupled heat-exchanger case.

Used by plot_energy.py and animate.py: case directories, plot style, the legacy-VTK
reader for the solver snapshots, the energy logs and the temperature-field panel.
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# palette (validated categorical order) and text inks
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET = "#4a3aa7"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MM = 1e3  # m -> mm for the axes


def case_dirs(argv):
    """(case dir, fluid output dir, solid output dir); argv[1] overrides the case dir."""
    case = argv[1] if len(argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    return case, os.path.join(case, "fluid", "output"), os.path.join(case, "solid", "output")


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 10, "legend.frameon": False,
    })


def temperature_cmap():
    """Diverging cold-hot map; masked (NaN) solid nodes are drawn dark grey."""
    cmap = LinearSegmentedColormap.from_list("cold-hot", [BLUE, "#e9e8e4", ORANGE])
    cmap.set_bad("#3d3d3a")
    return cmap


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def load_energy_logs(fluid_dir, solid_dir):
    """Both energy logs, truncated to the windows present in both."""
    fe = read_csv(os.path.join(fluid_dir, "fluid_energy.csv"))
    se = read_csv(os.path.join(solid_dir, "solid_energy.csv"))
    n = min(len(fe), len(se))
    return fe[:n], se[:n]


def sorted_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files matching {pattern}")
    return files


class Field:
    """One STRUCTURED_POINTS snapshot: T (NaN on solid nodes), velocity (or None),
    extent of the node centres (x0, x1, y0, y1), node spacing dx, dy."""

    def __init__(self, path):
        with open(path) as f:
            lines = f.read().split("\n")
        nx, ny = (int(v) for v in lines[4].split()[1:3])
        ox, oy = (float(v) for v in lines[5].split()[1:3])
        self.dx, self.dy = (float(v) for v in lines[6].split()[1:3])
        n = nx * ny
        self.T, self.vel = None, None
        for k, line in enumerate(lines):
            if line.startswith("SCALARS T "):
                self.T = np.array(lines[k + 2:k + 2 + n], dtype=float).reshape(ny, nx)
            elif line.startswith("VECTORS velocity"):
                flat = np.array(" ".join(lines[k + 1:k + 1 + n]).split(), dtype=float)
                self.vel = flat.reshape(ny, nx, 3)[:, :, :2]
        self.extent = (ox, ox + (nx - 1) * self.dx, oy, oy + (ny - 1) * self.dy)

    @property
    def x_nodes(self):
        return self.extent[0] + np.arange(self.T.shape[1]) * self.dx

    def bulk_temperature(self):
        """Mixing-cup temperature sum(u T)/sum(u) per column; the column flow rate is the
        same everywhere (mass conservation), so this is defined even with recirculation."""
        u = self.vel[:, :, 0]
        return np.nansum(u * np.nan_to_num(self.T), axis=0) / np.sum(u, axis=0)


def draw_fields(ax, fields, cmap, vmin, vmax, x_max):
    """Temperature of the given snapshots as cell images, streamlines where velocity is
    available, and the interface lines of the last (solid) field. Returns the last image."""
    ax.grid(False)
    for fld in fields:
        e, hx, hy = fld.extent, fld.dx, fld.dy
        ext = [(e[0] - hx / 2) * MM, (e[1] + hx / 2) * MM, (e[2] - hy / 2) * MM, (e[3] + hy / 2) * MM]
        im = ax.imshow(fld.T, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
    for fld in fields:
        if fld.vel is None:
            continue
        # lattice node centres form a uniform grid; keep the columns inside [0, L]
        xs = fld.x_nodes * MM
        ys = fld.extent[2] * MM + np.arange(fld.T.shape[0]) * fld.dy * MM
        keep = (xs >= 0) & (xs <= x_max * MM)
        ax.streamplot(xs[keep], ys, fld.vel[:, keep, 0], fld.vel[:, keep, 1], density=(2.6, 1.1),
                      color=INK, linewidth=0.5, arrowsize=0.6)
    for yi in fields[-1].extent[2:]:
        ax.axhline(yi * MM, color=INK, lw=0.8, ls="--")
    ax.set_xlim(0, x_max * MM)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    return im
