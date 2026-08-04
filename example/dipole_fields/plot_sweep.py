import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, hsv_to_rgb

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx} \usepackage{bm}",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
})

input_file = "sweep.npz"
output_file = "complex_map.pdf"
brightness_radius = 0.20
brightness_gamma = 1.0
show_magnets = True
show_grid = False
show_gradient_lines = True
gradient_density = 1.2
gradient_linewidth = 1.0
gradient_arrowsize = 0.9

def make_complex_image(x_final, y_final, status):
    final_position = x_final + 1j * y_final
    phase = np.angle(final_position)
    radial_distance = np.abs(final_position)
    hue = (phase + np.pi) / (2.0 * np.pi)
    saturation = np.ones_like(radial_distance)
    brightness = np.clip(radial_distance / brightness_radius,0.0,1.0,)**brightness_gamma
    image = hsv_to_rgb(np.stack([hue, saturation, brightness],axis=-1,))
    image[status == 0] = np.array([1.0, 1.0, 1.0])
    image[status == -1] = np.array([0.5, 0.5, 0.5])
    return image


def make_phase_colormap():
    phase = np.linspace(-np.pi, np.pi, 512,)
    hue = (phase + np.pi) / (2.0 * np.pi)
    color = hsv_to_rgb(
        np.stack(
            [hue, np.ones_like(hue), np.ones_like(hue),], axis=-1,)
    )
    return ListedColormap(color, name="phase",)


def calculate_gradient_lines(x, y, x_final, y_final, status):
    radial_distance = np.sqrt(x_final**2 + y_final**2)
    radial_distance[status != 1] = np.nan
    gradient_y, gradient_x = np.gradient(radial_distance, y, x,)
    gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)
    valid = (np.isfinite(gradient_x) & np.isfinite(gradient_y) & (gradient_norm > 0.0))
    gradient_x[valid] /= gradient_norm[valid]
    gradient_y[valid] /= gradient_norm[valid]
    gradient_x[~valid] = np.nan
    gradient_y[~valid] = np.nan
    return (
        np.ma.masked_invalid(gradient_x),
        np.ma.masked_invalid(gradient_y),
    )


data = np.load(input_file)
x = data["x"]
y = data["y"]
status = data["status"]
x_final = data["x_final"]
y_final = data["y_final"]
plate_size_x = float(data["plate_size_x"])
plate_size_y = float(data["plate_size_y"])
dipole_positions = data["dipole_positions"]
dipole_moments = data["dipole_moments"]
image = make_complex_image(x_final, y_final, status,)

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7), layout="constrained")

ax1.tick_params(
    which='both',
    bottom=True,
    top=False,
    left=True,
    right=False,
    labelbottom=True,
    labelleft=True,
    labelright=False,
    labeltop=False,
)

ax1.imshow(
    image,
    origin="lower",
    extent=[
        x[0],
        x[-1],
        y[0],
        y[-1],
    ],
    interpolation="nearest",
    aspect="equal",
)

if show_gradient_lines:
    gradient_x, gradient_y = calculate_gradient_lines(
        x,
        y,
        x_final,
        y_final,
        status,
    )

    ax1.streamplot(
        x,
        y,
        gradient_x,
        gradient_y,
        color=(1.0, 1.0, 1.0, 0.55),
        density=gradient_density,
        linewidth=gradient_linewidth,
        arrowsize=gradient_arrowsize,
        zorder=4,
    )

if show_magnets:
    ax1.quiver(
        dipole_positions[:, 0],
        dipole_positions[:, 1],
        dipole_moments[:, 0],
        dipole_moments[:, 1],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=8.0,
        width=0.008,
        zorder=5,
    )

    ax1.scatter(
        dipole_positions[:, 0],
        dipole_positions[:, 1],
        color="black",
        marker="s",
        s=70,
        zorder=6,
    )

ax1.set_xlabel(r"$x_0 \, [\unit{m}]$")
ax1.set_ylabel(r"$y_0 \, [\unit{m}]$")
ax1.set_xlim([-plate_size_x / 2.0, plate_size_x / 2.0,])
ax1.set_ylim([-plate_size_y / 2.0, plate_size_y / 2.0,])
ax1.grid(show_grid, alpha=0.3)
phase_colormap = make_phase_colormap()
phase_map = ScalarMappable(norm=Normalize(vmin=-np.pi, vmax=np.pi,), cmap=phase_colormap,)
phase_map.set_array([])
phase_colorbar = fig1.colorbar(
    phase_map,
    ax=ax1,
    fraction=0.050,
    pad=0.025,
)

phase_colorbar.set_label(
    r"$\arg z$"
)

phase_colorbar.set_ticks([
    -np.pi,
    -np.pi / 2.0,
    0.0,
    np.pi / 2.0,
    np.pi,
])

phase_colorbar.set_ticklabels([
    r"$-\pi$",
    r"$-\frac{\pi}{2}$",
    r"$0$",
    r"$\frac{\pi}{2}$",
    r"$\pi$",
])

brightness_map = ScalarMappable(
    norm=Normalize(
        vmin=0.0,
        vmax=brightness_radius,
    ),
    cmap="gray",
)

brightness_map.set_array([])

brightness_colorbar = fig1.colorbar(
    brightness_map,
    ax=ax1,
    fraction=0.050,
    pad=0.105,
)

brightness_colorbar.set_label(
    r"$|z| \, [\unit{m}]$"
)

plt.savefig(
    output_file,
    dpi=300,
)

successful = np.sum(status == 1)
skipped = np.sum(status == 0)
failed = np.sum(status == -1)

print(f"\nSuccessful  = {successful}")
print(f"Skipped     = {skipped}")
print(f"Failed      = {failed}")
print(f"Input       = {input_file}")
print(f"Figure      = {output_file}")
