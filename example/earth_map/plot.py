import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import simutils as su
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from pathlib import Path
from matplotlib.ticker import FuncFormatter

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": "\\usepackage{siunitx}\n\\usepackage{bm}",
    "font.size": 32,
    "axes.titlesize": 32,
    "axes.labelsize": 32,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 32,
})

R = 0.005  # sphere radius
##############################################################################
# PARAMETERS
INPUT_FILE = Path("results/earth_results.npz")
OUTPUT_FILE = Path("results/earth_map.png")
MARKERS = [
    (53.550556, 9.993682),
    (-6.200000, 106.826944),
    (35.689444, 139.691667),
    (-35.279722, 149.128998),
]
FIGSIZE = (14, 7)
DPI = 300
COLORMAP = "seismic"
VALUE_NAME = "x"      # or "t"
##############################################################################


def load_data(filename):
    data = np.load(filename)
    lat = data["lat"]
    lon = data["lon"]
    values = data[VALUE_NAME]
    return lat, lon, values


def create_figure():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.05, 0.1, 0.55, 0.6], projection=ccrs.Robinson())
    ax.set_global()
    return fig, ax


def draw_map(ax):
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.3,
        alpha=0.5,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.xlabel_style = {"size": 24}
    gl.ylabel_style = {"size": 24}
    gl.xlocator = cticker.LongitudeLocator()
    gl.ylocator = cticker.LatitudeLocator()
    gl.xformatter = FuncFormatter(lambda x, pos: rf"${x:.0f}^\circ$")
    gl.yformatter = FuncFormatter(lambda y, pos: rf"${y:.0f}^\circ$")


def draw_markers(ax, markers):
    styles = ["o", "s", "^", "v"]
    for (lat, lon), m in zip(markers, styles):
        ax.scatter(
            lon,
            lat,
            transform=ccrs.PlateCarree(),
            marker=m,
            s=80,
            edgecolor="black",
            linewidth=1.0,
            zorder=10,
            color="yellow",
        )


def draw_data(ax, lat, lon, values):
    mesh = ax.pcolormesh(
        lon,
        lat,
        values,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=COLORMAP,
    )
    return mesh


def colored_line(ax, x, y, c, cmap, norm, lw=2.0, alpha=1.0, zorder=2):
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)

    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    c_seg = 0.5 * (c[:-1] + c[1:])

    lc = LineCollection(
        segs, cmap=cmap, norm=norm,
        linewidth=lw, alpha=alpha, zorder=zorder
    )
    lc.set_array(c_seg)
    ax.add_collection(lc)
    return lc


def black_line(ax, x, y, lw=2.0, alpha=1.0, zorder=2):
    x = np.asarray(x)
    y = np.asarray(y)

    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(
        segs,
        colors="black",
        linewidths=lw,
        alpha=alpha,
        zorder=zorder
    )
    ax.add_collection(lc)
    return lc


datasets = su.extract()
names = ["hamburg", "jakarta", "tokyo", "australia"]

# Marker style per pair
markers = {
    "hamburg": "o",   # filled circle
    "jakarta": "s",    # filled square
    "tokyo": "^",     # filled triangle
    "australia": "v",     # filled reversed triangle
}

all_O = []
all_mu = []
all_xy = []

for name in names:
    d = datasets[name]

    x = np.asarray(d.x)
    y = np.asarray(d.y)

    # New solver output names: magnetic moment in world coordinates.
    mu_x = np.asarray(d.mu_world_x)
    mu_y = np.asarray(d.mu_world_y)
    mu_z = np.asarray(d.mu_world_z)

    mu_norm = np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
    mu_norm_safe = np.where(mu_norm == 0, np.nan, mu_norm)

    point_x = x + R * mu_x / mu_norm_safe
    point_y = y + R * mu_y / mu_norm_safe

    Ox, Oy, Oz = d.Ox, d.Oy, d.Oz
    O_norm = np.sqrt(Ox**2 + Oy**2 + Oz**2)

    # Keep only 1.0-scale fluctuations for the mu-coloring
    mu_norm_plot = np.round(mu_norm, 0)

    all_O.append(O_norm[np.isfinite(O_norm)])
    all_mu.append(mu_norm_plot[np.isfinite(mu_norm_plot)])

    all_xy.append(np.column_stack([x * 100, y * 100]))
    all_xy.append(np.column_stack([point_x * 100, point_y * 100]))

all_O = np.concatenate(all_O)
all_mu = np.concatenate(all_mu)

norm_O = Normalize(vmin=np.nanmin(all_O), vmax=np.nanmax(all_O))
#norm_mu = Normalize(vmin=np.nanmin(all_mu), vmax=np.nanmax(all_mu))

cmap_O = plt.cm.viridis
#cmap_mu = plt.cm.plasma
fig = plt.figure(figsize=(19, 8), layout='tight') #9,7.5

ax_map = fig.add_axes([0.03, 0.12, 0.47, 0.76],
                      projection=ccrs.Robinson())

ax = fig.add_axes([0.66, 0.12, 0.35, 0.76]) #0,32

lat, lon, values = load_data(INPUT_FILE)
draw_map(ax_map)
mesh = draw_data(
    ax_map,
    lon=lon,
    lat=lat,
    values=values,
)
draw_markers(ax_map, MARKERS)

for name in names:
    d = datasets[name]

    x = np.asarray(d.x)
    y = np.asarray(d.y)

    # New solver output names: magnetic moment in world coordinates.
    mu_x = np.asarray(d.mu_world_x)
    mu_y = np.asarray(d.mu_world_y)
    mu_z = np.asarray(d.mu_world_z)

    mu_norm = np.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
    mu_norm_safe = np.where(mu_norm == 0, np.nan, mu_norm)
    mu_norm_plot = np.round(mu_norm, 0)

    point_x = x + R * mu_x / mu_norm_safe
    point_y = y + R * mu_y / mu_norm_safe

    Ox, Oy, Oz = d.Ox, d.Oy, d.Oz
    O_norm = np.sqrt(Ox**2 + Oy**2 + Oz**2)

    # COM trajectory
    colored_line(ax, x * 100, y * 100, O_norm, cmap_O, norm_O, lw=2.0)

    # mu-tip trajectory
    black_line(ax, point_x * 100, point_y * 100, lw=2.0)

    m = markers[name]

    # Start/end points for COM trajectory
    ax.scatter(
        x[0] * 100, y[0] * 100,
        marker=m, s=80, facecolors="white", edgecolors="black", linewidths=1.2, zorder=5
    )
    ax.scatter(
        x[-1] * 100, y[-1] * 100,
        marker=m, s=80, facecolors="black", edgecolors="black", linewidths=1.2, zorder=5
    )

    # Start/end points for mu-tip trajectory
    ax.scatter(
        point_x[0] * 100, point_y[0] * 100,
        marker=m, s=80, facecolors="white", edgecolors="black", linewidths=1.2, zorder=5
    )
    ax.scatter(
        point_x[-1] * 100, point_y[-1] * 100,
        marker=m, s=80, facecolors="black", edgecolors="black", linewidths=1.2, zorder=5
    )

from matplotlib.lines import Line2D

shape_legend = [
    Line2D([0], [0], marker='o', linestyle='None',
           markerfacecolor='black', markeredgecolor='black',
           markersize=10, label=r"$\textrm{Hamburg}$"),
    Line2D([0], [0], marker='s', linestyle='None',
           markerfacecolor='black', markeredgecolor='black',
           markersize=10, label=r"$\textrm{Jakarta}$"),
    Line2D([0], [0], marker='^', linestyle='None',
           markerfacecolor='black', markeredgecolor='black',
           markersize=10, label=r"$\textrm{Tokyo}$"),
    Line2D([0], [0], marker='v', linestyle='None',
           markerfacecolor='black', markeredgecolor='black',
           markersize=10, label=r"$\textrm{Canberra}$"),
]

ax.legend(
    handles=shape_legend,
    loc="upper right",
    handlelength=0.6,
    handletextpad=0.4,
    frameon=True,
    borderaxespad=0.3
    )
all_xy = np.concatenate(all_xy, axis=0)
xmin, ymin = np.nanmin(all_xy, axis=0)
xmax, ymax = np.nanmax(all_xy, axis=0)

pad_x = 0.10 * (xmax - xmin)
pad_y = 0.05 * (ymax - ymin)

# Program coordinates are +x west and +y south, therefore both display
# directions are reversed for the conventional east-right/north-up view.
ax.set_xlim(xmax + pad_x, xmin - pad_x - 0.3)
ax.set_ylim(ymax + pad_y, ymin - pad_y)

ax.set_xlabel(r"$x \, [\unit{\centi\metre}]$")
ax.set_ylabel(r"$y \, [\unit{\centi\metre}]$")
ax.grid(True, alpha=0.3)

sm_O = ScalarMappable(norm=norm_O, cmap=cmap_O)
sm_O.set_array([])
cbar1 = fig.colorbar(sm_O, ax=ax, fraction=0.046, pad=0.04)
cbar1.set_label(r"$\| \bm{\Omega} \| \, [\unit{\second^{-1}}]$")
cbar1.set_ticks([0, 50, 100, 150, 200])
cbar2 = fig.colorbar(mesh, ax=ax_map, fraction=0.046, pad=0.04)
cbar2.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${100*x:.1f}$"))
cbar2.set_label(r"$x|_{t=0.5 \, \unit{s}} \, [\unit{cm}]$")
cbar2.set_ticks([-0.08, -0.04, 0.0, 0.08, 0.04])
ax_map.text(
    0.02,
    1.00,
    "(a)",
    transform=ax_map.transAxes,
    ha="left",
    va="top",
    fontsize=25,
    fontweight="bold",
)
ax.text(
    0.02,
    0.98,
    "(b)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=25,
    fontweight="bold",
)
plt.savefig("combined.png", dpi=300, bbox_inches="tight")
plt.savefig("combined.pdf", dpi=300, bbox_inches="tight")
