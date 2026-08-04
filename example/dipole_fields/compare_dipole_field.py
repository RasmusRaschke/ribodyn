import csv
import subprocess
import tempfile
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 

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

base_directory = Path(__file__).resolve().parent
solver = (
    base_directory
    / "../../build/solver"
).resolve()
output_file = "dipole_field_comparison.pdf"
plate_size_x = 0.30
plate_size_y = 0.30
n_grid = 250
stream_density = 2.0
z_plane = 0.0
probe_position = np.array([
    0.000,
    0.000,
    z_plane,
])
probe_dt = 1.0e-6
probe_t_end = 1.0e-6

dipole_field_scale = 1.0e-7 #SI
#Dipoles are divergent at their centre, so you have to exclude a small neighbourhood
dipole_minimum_distance = 0.006
magnet_exclusion_radius = 0.001

# Position in m, dipole Am^2
dipole_positions = np.array([
    [0.1050, 0.0000, 0.0050],
    [-0.0525, 0.0909326674, 0.0050],
    [-0.0525, -0.0909326674, 0.0050],
])

dipole_moments = np.array([
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
])

#Calculate Field manually
def calculate_field(position):
    position = np.asarray(position)

    field = np.zeros(
        position.shape,
        dtype=float,
    )

    for dipole_position, dipole_moment in zip(
        dipole_positions,
        dipole_moments,
    ):
        displacement = (
            position
            - dipole_position
        )

        radius = np.linalg.norm(
            displacement,
            axis=-1,
        )

        projection = np.sum(
            dipole_moment
            * displacement,
            axis=-1,
        )

        field += dipole_field_scale * (
            3.0
            * displacement
            * projection[..., np.newaxis]
            / radius[..., np.newaxis]**5
            - dipole_moment
            / radius[..., np.newaxis]**3
        )

    return field


def calculate_magnet_mask(x, y):
    mask = np.zeros_like(
        x,
        dtype=bool,
    )

    for position in dipole_positions:
        distance = np.sqrt(
            (x - position[0])**2
            + (y - position[1])**2
        )

        mask |= (
            distance
            < magnet_exclusion_radius
        )

    return mask

#Extract field of solver for a test case with non-magnetic probe

def make_probe_input():
    lines = [
        "mass 1.0",
        "radius 0.01",
        "charge 0.0",
        "magneticMoment 0.0 0.0 0.0",
        "magneticPolarizability 0 0 0  0 0 0  0 0 0",
        "inertia 1 0 0  0 1 0  0 0 1",
        "",
        (
            "position "
            f"{probe_position[0]:.17g} "
            f"{probe_position[1]:.17g} "
            f"{probe_position[2]:.17g}"
        ),
        "velocity 0.0 0.0 0.0",
        "omega 0.0 0.0 0.0",
        "quaternion 1.0 0.0 0.0 0.0",
        "",
        "solverMode dAlembert",
        "",
        "constraint none",
        "",
        "gravityType none",
        "",
        "emType magDipole",
        f"dipoleCount {len(dipole_positions)}",
    ]

    for index, (position, moment) in enumerate(
        zip(dipole_positions, dipole_moments),
        start=1,
    ):
        lines.append(
            f"dipolePosition{index} "
            f"{position[0]:.17g} "
            f"{position[1]:.17g} "
            f"{position[2]:.17g}"
        )

        lines.append(
            f"dipoleMoment{index} "
            f"{moment[0]:.17g} "
            f"{moment[1]:.17g} "
            f"{moment[2]:.17g}"
        )

    lines.extend([
        f"dipoleFieldScale {dipole_field_scale:.17g}",
        f"dipoleMinimumDistance {dipole_minimum_distance:.17g}",
        "",
        "airType none",
        "rollingResistanceType none",
        "",
        f"dt {probe_dt:.17g}",
        f"tEnd {probe_t_end:.17g}",
        "",
    ])

    return "\n".join(lines)


def read_final_field(filename):
    with open(filename, "r", newline="") as file:
        reader = csv.DictReader(file)
        final_row = None

        for row in reader:
            final_row = row

    if final_row is None:
        raise RuntimeError(
            "The C++ program produced no output rows."
        )

    return np.array([
        float(final_row["Bx_world"]),
        float(final_row["By_world"]),
        float(final_row["Bz_world"]),
    ])


def calculate_cpp_field():
    if not solver.is_file():
        raise FileNotFoundError(
            f"Solver not found:\n{solver}"
        )

    with tempfile.TemporaryDirectory(
        prefix="ribodyn_field_probe_"
    ) as workdir:
        input_name = Path(workdir) / "input.in"
        output_name = Path(workdir) / "output.csv"

        with open(input_name, "w") as file:
            file.write(
                make_probe_input()
            )

        process = subprocess.run(
            [
                str(solver),
                str(input_name),
            ],
            cwd=workdir,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(
                "The C++ probe failed.\n\n"
                f"stdout:\n{process.stdout}\n\n"
                f"stderr:\n{process.stderr}"
            )

        return read_final_field(
            output_name
        )


# Grid for statistics and plotting

x = np.linspace(
    -plate_size_x / 2.0,
    plate_size_x / 2.0,
    n_grid,
)

y = np.linspace(
    -plate_size_y / 2.0,
    plate_size_y / 2.0,
    n_grid,
)

xx, yy = np.meshgrid(x, y)

positions = np.stack(
    [
        xx,
        yy,
        np.full_like(xx, z_plane),
    ],
    axis=-1,
)

field_grid = calculate_field(positions)
magnet_mask = calculate_magnet_mask(xx, yy,)
field_grid[magnet_mask] = np.nan
field_magnitude = np.linalg.norm(field_grid, axis=-1,)
log_field_magnitude = np.log10(field_magnitude)
field_manual = calculate_field(probe_position)
field_cpp = calculate_cpp_field()
difference = (field_manual - field_cpp)
absolute_error = np.linalg.norm(difference)
relative_error = (absolute_error / np.linalg.norm(field_manual))

####################################################################
print("\nProbe position")
print("--------------")
print(
    f"x = {probe_position[0]:.6e} m"
)
print(
    f"y = {probe_position[1]:.6e} m"
)
print(
    f"z = {probe_position[2]:.6e} m"
)

print("\nManual field")
print("------------")
print(
    f"Bx = {field_manual[0]:.12e} T"
)
print(
    f"By = {field_manual[1]:.12e} T"
)
print(
    f"Bz = {field_manual[2]:.12e} T"
)

print("\nC++ field")
print("---------")
print(
    f"Bx = {field_cpp[0]:.12e} T"
)
print(
    f"By = {field_cpp[1]:.12e} T"
)
print(
    f"Bz = {field_cpp[2]:.12e} T"
)

print("\nError")
print("-----")
print(
    f"Absolute = {absolute_error:.12e} T"
)
print(
    f"Relative = {relative_error:.12e}"
)


fig1, ax1 = plt.subplots(figsize=(15, 15), layout="constrained")

stream = ax1.streamplot(
    x,
    y,
    field_grid[:, :, 0],
    field_grid[:, :, 1],
    color=log_field_magnitude,
    cmap="viridis",
    density=stream_density,
    linewidth=1.8,
    arrowsize=1.4,
)

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
    label=r"$\textrm{dipoles}$",
    zorder=6,
)

ax1.scatter(
    probe_position[0],
    probe_position[1],
    color="red",
    marker="x",
    s=100,
    linewidths=3,
    label=r"$\textrm{probe}$",
    zorder=7,
)
ax1.set_aspect("equal")
ax1.set_xlabel(r"$x \, [\unit{m}]$")
ax1.set_ylabel(r"$y \, [\unit{m}]$")

ax1.set_xlim([
    -plate_size_x / 2.0,
    plate_size_x / 2.0,
])

ax1.set_ylim([
    -plate_size_y / 2.0,
    plate_size_y / 2.0,
])

ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

ax1.text(
    0.01,
    0.98,
    "(a)",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=20,
    fontweight="bold",
)

colorbar = fig1.colorbar(
    stream.lines,
    #ax=ax1,
)

colorbar.set_label(
    r"$\log_{10}|\bm{B}|$"
)

plt.savefig(
    output_file,
    dpi=300,
)

print(f"\nFigure = {output_file}")