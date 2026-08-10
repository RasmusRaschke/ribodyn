import csv
import subprocess
import tempfile
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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

### Settings ###

base_directory = Path(__file__).resolve().parent

solver = (
    base_directory
    / "../../../build/solver"
).resolve()

output_file = "full_field.pdf"

progress_every = 100

plate_size_x = 0.8
plate_size_y = 0.8
z_min_3d = -0.30
z_max_3d = 0.30
n_grid_2d = 80
z_plane_2d = 0.000
stream_density = 1.5
n_grid_x_3d = 20
n_grid_y_3d = 20
n_grid_z_3d = 20



field_line_seed_count_x = 4
field_line_seed_count_y = 4
field_line_seed_count_z = 5

field_line_seed_margin_x = 0.020
field_line_seed_margin_y = 0.020
field_line_seed_margin_z = 0.015

field_line_step = 0.004
field_line_steps = 250
field_line_linewidth = 1.5
field_line_arrow_count = 6
field_line_arrow_length = 0.030

show_plane_z0 = True
plane_alpha = 0.30

dipole_field_scale = 1.0e-7
dipole_minimum_distance = 0.0005
magnet_exclusion_radius = 0.001

probe_dt = 1.0e-6
probe_t_end = 1.0e-6

# Position in m, dipole moment in A m^2.
dipole_positions = np.array([
    [0.3, 0.0000, 0.1],
    [0.1, 0.0000, 0.1],
    [-0.1, 0.0000, 0.1],
    [-0.3, 0.0000, 0.1],
    #[-0.1, -0.166, 0.0050],
    #[-0.1, 0.166, 0.0050],
])

dipole_moments = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
])


#################################################################

def make_probe_input(position):
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
            f"{position[0]:.17g} "
            f"{position[1]:.17g} "
            f"{position[2]:.17g}"
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

    for index, (position_dipole, moment_dipole) in enumerate(
        zip(dipole_positions, dipole_moments),
        start=1,
    ):
        lines.append(
            f"dipolePosition{index} "
            f"{position_dipole[0]:.17g} "
            f"{position_dipole[1]:.17g} "
            f"{position_dipole[2]:.17g}"
        )

        lines.append(
            f"dipoleMoment{index} "
            f"{moment_dipole[0]:.17g} "
            f"{moment_dipole[1]:.17g} "
            f"{moment_dipole[2]:.17g}"
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


def point_is_inside_magnet(position):
    distance = np.sqrt(
        (position[0] - dipole_positions[:, 0])**2
        + (position[1] - dipole_positions[:, 1])**2
        + (position[2] - dipole_positions[:, 2])**2
    )

    return np.any(
        distance < magnet_exclusion_radius
    )


def run_probe(task):
    index, position = task

    if point_is_inside_magnet(position):
        return index, 0, np.full(3, np.nan)

    with tempfile.TemporaryDirectory(
        prefix="ribodyn_field_probe_"
    ) as workdir:
        input_name = Path(workdir) / "input.in"
        output_name = Path(workdir) / "output.csv"

        with open(input_name, "w") as file:
            file.write(
                make_probe_input(position)
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
            print(
                f"\nProbe failed at "
                f"x = {position[0]:.6e}, "
                f"y = {position[1]:.6e}, "
                f"z = {position[2]:.6e}"
            )
            print(process.stderr)

            return index, -1, np.full(3, np.nan)

        field = read_final_field(output_name)

    return index, 1, field


# =============================================================================
# Field-line interpolation
# =============================================================================

def interpolate_field(position, x, y, z, field):
    x_position = position[0]
    y_position = position[1]
    z_position = position[2]

    if (
        x_position < x[0]
        or x_position > x[-1]
        or y_position < y[0]
        or y_position > y[-1]
        or z_position < z[0]
        or z_position > z[-1]
    ):
        return None

    x_upper = np.searchsorted(x, x_position, side="right")
    y_upper = np.searchsorted(y, y_position, side="right")
    z_upper = np.searchsorted(z, z_position, side="right")

    x_upper = min(max(x_upper, 1), len(x) - 1)
    y_upper = min(max(y_upper, 1), len(y) - 1)
    z_upper = min(max(z_upper, 1), len(z) - 1)

    x_lower = x_upper - 1
    y_lower = y_upper - 1
    z_lower = z_upper - 1

    tx = (
        x_position - x[x_lower]
    ) / (
        x[x_upper] - x[x_lower]
    )

    ty = (
        y_position - y[y_lower]
    ) / (
        y[y_upper] - y[y_lower]
    )

    tz = (
        z_position - z[z_lower]
    ) / (
        z[z_upper] - z[z_lower]
    )

    c000 = field[x_lower, y_lower, z_lower]
    c100 = field[x_upper, y_lower, z_lower]
    c010 = field[x_lower, y_upper, z_lower]
    c110 = field[x_upper, y_upper, z_lower]
    c001 = field[x_lower, y_lower, z_upper]
    c101 = field[x_upper, y_lower, z_upper]
    c011 = field[x_lower, y_upper, z_upper]
    c111 = field[x_upper, y_upper, z_upper]

    corners = np.array([
        c000,
        c100,
        c010,
        c110,
        c001,
        c101,
        c011,
        c111,
    ])

    if not np.isfinite(corners).all():
        return None

    c00 = (1.0 - tx) * c000 + tx * c100
    c10 = (1.0 - tx) * c010 + tx * c110
    c01 = (1.0 - tx) * c001 + tx * c101
    c11 = (1.0 - tx) * c011 + tx * c111

    c0 = (1.0 - ty) * c00 + ty * c10
    c1 = (1.0 - ty) * c01 + ty * c11

    return (1.0 - tz) * c0 + tz * c1


def field_direction(position, x, y, z, field):
    field_value = interpolate_field(
        position,
        x,
        y,
        z,
        field,
    )

    if field_value is None:
        return None

    field_magnitude = np.linalg.norm(field_value)

    if (
        not np.isfinite(field_magnitude)
        or field_magnitude <= 0.0
    ):
        return None

    return field_value / field_magnitude


def inside_field_domain(position, x, y, z):
    return (
        x[0] <= position[0] <= x[-1]
        and y[0] <= position[1] <= y[-1]
        and z[0] <= position[2] <= z[-1]
    )


def field_line_step_rk4(
    position,
    direction_sign,
    step_size,
    x,
    y,
    z,
    field,
):
    k1 = field_direction(position, x, y, z, field)

    if k1 is None:
        return None

    k2 = field_direction(
        position
        + 0.5
        * direction_sign
        * step_size
        * k1,
        x,
        y,
        z,
        field,
    )

    if k2 is None:
        return None

    k3 = field_direction(
        position
        + 0.5
        * direction_sign
        * step_size
        * k2,
        x,
        y,
        z,
        field,
    )

    if k3 is None:
        return None

    k4 = field_direction(
        position
        + direction_sign
        * step_size
        * k3,
        x,
        y,
        z,
        field,
    )

    if k4 is None:
        return None

    return (
        position
        + direction_sign
        * step_size
        / 6.0
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )


def trace_field_line(
    seed,
    direction_sign,
    x,
    y,
    z,
    field,
):
    points = [
        np.array(
            seed,
            dtype=float,
        )
    ]

    position = np.array(
        seed,
        dtype=float,
    )

    for _ in range(field_line_steps):
        next_position = field_line_step_rk4(
            position,
            direction_sign,
            field_line_step,
            x,
            y,
            z,
            field,
        )

        if next_position is None:
            break

        if not inside_field_domain(
            next_position,
            x,
            y,
            z,
        ):
            break

        if point_is_inside_magnet(next_position):
            break

        points.append(next_position.copy())
        position = next_position

    return np.array(points)


def build_field_line(seed, x, y, z, field):
    backward = trace_field_line(
        seed,
        -1.0,
        x,
        y,
        z,
        field,
    )

    forward = trace_field_line(
        seed,
        1.0,
        x,
        y,
        z,
        field,
    )

    if len(backward) > 0:
        backward = backward[::-1][:-1]

    line = np.concatenate(
        [
            backward,
            forward,
        ],
        axis=0,
    )

    if len(line) < 2:
        return None

    return line


def make_field_line_seeds():
    seed_x = np.linspace(
        -plate_size_x / 2.0
        + field_line_seed_margin_x,
        plate_size_x / 2.0
        - field_line_seed_margin_x,
        field_line_seed_count_x,
    )

    seed_y = np.linspace(
        -plate_size_y / 2.0
        + field_line_seed_margin_y,
        plate_size_y / 2.0
        - field_line_seed_margin_y,
        field_line_seed_count_y,
    )

    seed_z = np.linspace(
        z_min_3d
        + field_line_seed_margin_z,
        z_max_3d
        - field_line_seed_margin_z,
        field_line_seed_count_z,
    )

    seeds = []

    for x_position in seed_x:
        for y_position in seed_y:
            for z_position in seed_z:
                seed = np.array([
                    x_position,
                    y_position,
                    z_position,
                ])

                if not point_is_inside_magnet(seed):
                    seeds.append(seed)

    return seeds


# =============================================================================
# Build grids
# =============================================================================

x_2d = np.linspace(
    -plate_size_x / 2.0,
    plate_size_x / 2.0,
    n_grid_2d,
)

y_2d = np.linspace(
    -plate_size_y / 2.0,
    plate_size_y / 2.0,
    n_grid_2d,
)

xx_2d, yy_2d = np.meshgrid(
    x_2d,
    y_2d,
)

positions_2d = np.stack(
    [
        xx_2d,
        yy_2d,
        np.full_like(xx_2d, z_plane_2d),
    ],
    axis=-1,
)

x_3d = np.linspace(
    -plate_size_x / 2.0,
    plate_size_x / 2.0,
    n_grid_x_3d,
)

y_3d = np.linspace(
    -plate_size_y / 2.0,
    plate_size_y / 2.0,
    n_grid_y_3d,
)

z_3d = np.linspace(
    z_min_3d,
    z_max_3d,
    n_grid_z_3d,
)

xx_3d, yy_3d, zz_3d = np.meshgrid(
    x_3d,
    y_3d,
    z_3d,
    indexing="ij",
)

positions_3d = np.stack(
    [
        xx_3d,
        yy_3d,
        zz_3d,
    ],
    axis=-1,
)

tasks_2d = [
    (
        index,
        positions_2d.reshape(-1, 3)[index],
    )
    for index in range(positions_2d.shape[0] * positions_2d.shape[1])
]

tasks_3d = [
    (
        index,
        positions_3d.reshape(-1, 3)[index],
    )
    for index in range(
        positions_3d.shape[0]
        * positions_3d.shape[1]
        * positions_3d.shape[2]
    )
]

all_tasks = (
    [("2d", task) for task in tasks_2d]
    + [("3d", task) for task in tasks_3d]
)


def run_labeled_probe(item):
    label, task = item
    index, status, field = run_probe(task)
    return label, index, status, field


# =============================================================================
# Execute probes
# =============================================================================

if not solver.is_file():
    raise FileNotFoundError(
        f"Solver not found:\n{solver}"
    )

print(f"\n2D points       = {len(tasks_2d)}")
print(f"3D points       = {len(tasks_3d)}")
print(f"Total probes    = {len(all_tasks)}")
print("Execution       = sequential")

status_2d = np.zeros(len(tasks_2d), dtype=int)
field_2d = np.full((len(tasks_2d), 3), np.nan)

status_3d = np.zeros(len(tasks_3d), dtype=int)
field_3d = np.full((len(tasks_3d), 3), np.nan)

successful = 0
skipped = 0
failed = 0

for calculation, item in enumerate(
    all_tasks,
    start=1,
):
    label, task = item
    index, point_status, point_field = run_probe(
        task
    )

    if label == "2d":
        status_2d[index] = point_status
        field_2d[index] = point_field
    else:
        status_3d[index] = point_status
        field_3d[index] = point_field

    if point_status == 1:
        successful += 1
    elif point_status == 0:
        skipped += 1
    else:
        failed += 1

    if (
        calculation % progress_every == 0
        or calculation == len(all_tasks)
    ):
        print(
            f"Finished {calculation} / {len(all_tasks)} "
            f"({100.0 * calculation / len(all_tasks):.1f} %), "
            f"successful = {successful}, "
            f"skipped = {skipped}, "
            f"failed = {failed}",
            flush=True,
        )

field_2d = field_2d.reshape(
    (n_grid_2d, n_grid_2d, 3)
)

status_2d = status_2d.reshape(
    (n_grid_2d, n_grid_2d)
)

field_3d = field_3d.reshape(
    (
        n_grid_x_3d,
        n_grid_y_3d,
        n_grid_z_3d,
        3,
    )
)

status_3d = status_3d.reshape(
    (
        n_grid_x_3d,
        n_grid_y_3d,
        n_grid_z_3d,
    )
)

field_magnitude_2d = np.linalg.norm(
    field_2d,
    axis=-1,
)

log_field_magnitude_2d = np.log10(
    field_magnitude_2d
)

field_magnitude_3d = np.linalg.norm(
    field_3d,
    axis=-1,
)

with np.errstate(
    divide="ignore",
    invalid="ignore",
):
    log_field_magnitude_3d = np.log10(
        field_magnitude_3d
    )

valid_3d = (
    (status_3d == 1)
    & np.isfinite(log_field_magnitude_3d)
)

print(f"\nSuccessful  = {successful}")
print(f"Skipped     = {skipped}")
print(f"Failed      = {failed}")


# =============================================================================
# Plot
# =============================================================================

fig1 = plt.figure(
    figsize=(22, 10),
    layout="constrained",
)

ax1 = fig1.add_subplot(
    1,
    2,
    1,
)

ax2 = fig1.add_subplot(
    1,
    2,
    2,
    projection="3d",
)

stream = ax1.streamplot(
    x_2d,
    y_2d,
    field_2d[:, :, 0],
    field_2d[:, :, 1],
    color=log_field_magnitude_2d,
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

ax1.set_aspect("equal")
ax1.set_xlabel(r"$x \, [\unit{m}]$")
ax1.set_ylabel(r"$y \, [\unit{m}]$")

# Internal frame:
#
#     +x = west
#     +y = south
#     +z = up
#
# Therefore positive x must point left and positive y must point down
# in the usual north-up/east-right page view.
ax1.set_xlim([
    plate_size_x / 2.0,
    -plate_size_x / 2.0,
])

ax1.set_ylim([
    plate_size_y / 2.0,
    -plate_size_y / 2.0,
])

ax1.grid(True, alpha=0.3)

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

colorbar_2d = fig1.colorbar(
    stream.lines,
    ax=ax1,
    fraction=0.050,
    pad=0.025,
)

colorbar_2d.set_label(
    r"$\log_{10}|\bm{B}|$"
)

field_norm_3d = Normalize(
    vmin=np.nanmin(
        log_field_magnitude_3d[
            valid_3d
        ]
    ),
    vmax=np.nanmax(
        log_field_magnitude_3d[
            valid_3d
        ]
    ),
)

field_colormap = plt.get_cmap(
    "viridis"
)

field_line_seeds = make_field_line_seeds()

for seed in field_line_seeds:
    line = build_field_line(
        seed,
        x_3d,
        y_3d,
        z_3d,
        field_3d,
    )

    if line is None:
        continue

    line_fields = []

    for position in line:
        field_value = interpolate_field(
            position,
            x_3d,
            y_3d,
            z_3d,
            field_3d,
        )

        if field_value is None:
            line_fields.append(
                np.full(3, np.nan)
            )
        else:
            line_fields.append(field_value)

    line_fields = np.array(line_fields)

    line_magnitude = np.linalg.norm(
        line_fields,
        axis=-1,
    )

    valid_line = np.isfinite(line_magnitude)

    line = line[valid_line]
    line_magnitude = line_magnitude[valid_line]

    if len(line) < 2:
        continue

    line_color_value = np.mean(
        np.log10(line_magnitude)
    )

    line_color = field_colormap(
        field_norm_3d(line_color_value)
    )

    ax2.plot(
        line[:, 0],
        line[:, 1],
        line[:, 2],
        color=line_color,
        linewidth=field_line_linewidth,
    )

    if len(line) > 2:
        arrow_indices = np.linspace(
            1,
            len(line) - 2,
            field_line_arrow_count,
            dtype=int,
        )

        arrow_indices = np.unique(
            arrow_indices
        )

        for arrow_index in arrow_indices:
            direction = (
                line[arrow_index + 1]
                - line[arrow_index - 1]
            )

            direction_magnitude = np.linalg.norm(
                direction
            )

            if direction_magnitude <= 0.0:
                continue

            direction /= direction_magnitude

            ax2.quiver(
                line[arrow_index, 0],
                line[arrow_index, 1],
                line[arrow_index, 2],
                direction[0],
                direction[1],
                direction[2],
                length=field_line_arrow_length,
                normalize=False,
                color=line_color,
                linewidth=1.0,
                arrow_length_ratio=0.35,
            )

if show_plane_z0:
    plane_x, plane_y = np.meshgrid(
        np.linspace(
            -plate_size_x / 2.0,
            plate_size_x / 2.0,
            2,
        ),
        np.linspace(
            -plate_size_y / 2.0,
            plate_size_y / 2.0,
            2,
        ),
    )

    plane_z = np.zeros_like(plane_x)

    ax2.plot_surface(
        plane_x,
        plane_y,
        plane_z,
        color="gray",
        alpha=plane_alpha,
        linewidth=0.0,
        shade=False,
        zorder=0,
    )

ax2.scatter(
    dipole_positions[:, 0],
    dipole_positions[:, 1],
    dipole_positions[:, 2],
    color="black",
    marker="s",
    s=45,
)

ax2.quiver(
    dipole_positions[:, 0],
    dipole_positions[:, 1],
    dipole_positions[:, 2],
    dipole_moments[:, 0],
    dipole_moments[:, 1],
    dipole_moments[:, 2],
    color="black",
    length=field_line_arrow_length,
    normalize=True,
    linewidth=2.0,
    arrow_length_ratio=0.30,
)

ax2.set_xlabel(r"$x \, [\unit{m}]$")
ax2.set_ylabel(r"$y \, [\unit{m}]$")
ax2.set_zlabel(r"$z \, [\unit{m}]$")

ax2.set_xlim([
    plate_size_x / 2.0,
    -plate_size_x / 2.0,
])

ax2.set_ylim([
    plate_size_y / 2.0,
    -plate_size_y / 2.0,
])

ax2.set_zlim([
    z_min_3d,
    z_max_3d,
])

ax2.set_box_aspect((
    plate_size_x,
    plate_size_y,
    z_max_3d - z_min_3d,
))

ax2.view_init(
    elev=25,
    azim=-45,
)

ax2.text2D(
    0.01,
    0.98,
    "(b)",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=20,
    fontweight="bold",
)

field_map_3d = ScalarMappable(
    norm=field_norm_3d,
    cmap="viridis",
)

field_map_3d.set_array([])

plt.savefig(
    output_file,
    dpi=300,
)

print(f"\nFigure = {output_file}")
