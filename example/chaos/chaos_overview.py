import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
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


##############################################################################
input_file = "earth_results.npz"
output_file = "stretching_overview.pdf"
candidate_file = "reference_candidates.txt"
combination_index = 0
number_of_candidates = 10
# Make tangent vectors dimensionless by appropriate scaling
position_scale = 0.01
angle_scale = 1.0
velocity_scale = 0.1
omega_scale = 100.0
##############################################################################


def normalize_quaternion(q):
    return q / np.linalg.norm(q)


def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def relative_rotation_vector(q_ref, q_other):
    q_ref = normalize_quaternion(q_ref)
    q_other = normalize_quaternion(q_other)
    q_rel = quaternion_multiply(quaternion_conjugate(q_ref), q_other)
    q_rel = normalize_quaternion(q_rel)
    if q_rel[0] < 0.0:
        q_rel = -q_rel
    vector = q_rel[1:]
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-14:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(vector_norm, q_rel[0])
    return angle * vector / vector_norm


def quaternion_grid(data):
    return np.stack([
        data["qw_final"][combination_index],
        data["qx_final"][combination_index],
        data["qy_final"][combination_index],
        data["qz_final"][combination_index],
    ], axis=-1)


def derivative(array, coordinates, axis):
    return np.gradient(array, coordinates, axis=axis, edge_order=2)


def orientation_derivatives(q, x, y, status):
    n_y, n_x = status.shape
    dtheta_dx = np.full((n_y, n_x, 3), np.nan)
    dtheta_dy = np.full_like(dtheta_dx, np.nan)
    for i in range(n_y):
        for j in range(n_x):
            if status[i, j] != 1:
                continue
            q_center = q[i, j]
            if (j > 0 and j < n_x - 1 and status[i, j - 1] == 1 and status[i, j + 1] == 1):
                minus = relative_rotation_vector(q_center, q[i, j - 1])
                plus = relative_rotation_vector(q_center, q[i, j + 1])
                dtheta_dx[i, j] = (plus - minus) / (x[j + 1] - x[j - 1])
            elif (j < n_x - 1 and status[i, j + 1] == 1):
                dtheta_dx[i, j] = (relative_rotation_vector(q_center, q[i, j + 1]) / (x[j + 1] - x[j]))
            elif (j > 0 and status[i, j - 1] == 1):
                dtheta_dx[i, j] = (-relative_rotation_vector(q_center, q[i, j - 1]) / (x[j] - x[j - 1]))
            if (i > 0 and i < n_y - 1 and status[i - 1, j] == 1 and status[i + 1, j] == 1):
                minus = relative_rotation_vector(q_center, q[i - 1, j])
                plus = relative_rotation_vector(q_center, q[i + 1, j])
                dtheta_dy[i, j] = (plus - minus) / (y[i + 1] - y[i - 1])
            elif (i < n_y - 1 and status[i + 1, j] == 1):
                dtheta_dy[i, j] = (relative_rotation_vector(q_center, q[i + 1, j]) / (y[i + 1] - y[i]))
            elif (i > 0 and status[i - 1, j] == 1):
                dtheta_dy[i, j] = (-relative_rotation_vector(q_center, q[i - 1, j]) / (y[i] - y[i - 1]))
    return dtheta_dx, dtheta_dy


def make_full_state(data):
    r = np.stack([
        data["x_final"][combination_index],
        data["y_final"][combination_index],
        data["z_final"][combination_index],
    ], axis=-1)
    v = np.stack([
        data["vx_final"][combination_index],
        data["vy_final"][combination_index],
        data["vz_final"][combination_index],
    ], axis=-1)
    Omega = np.stack([
        data["Ox_final"][combination_index],
        data["Oy_final"][combination_index],
        data["Oz_final"][combination_index],
    ], axis=-1)
    q = quaternion_grid(data)
    return r, q, v, Omega


def projected_jacobian(data):
    x = data["x"]
    y = data["y"]
    status = data["status"][combination_index]
    r, q, v, Omega = make_full_state(data)
    dr_dx = derivative(r, x, axis=1)
    dr_dy = derivative(r, y, axis=0)
    dv_dx = derivative(v, x, axis=1)
    dv_dy = derivative(v, y, axis=0)
    dOmega_dx = derivative(Omega, x, axis=1)
    dOmega_dy = derivative(Omega, y, axis=0)
    dtheta_dx, dtheta_dy = (orientation_derivatives(q, x, y, status))
    Jx = np.concatenate([dr_dx / position_scale, dtheta_dx / angle_scale, dv_dx / velocity_scale, dOmega_dx / omega_scale], axis=-1)
    Jy = np.concatenate([dr_dy / position_scale, dtheta_dy / angle_scale, dv_dy / velocity_scale, dOmega_dy / omega_scale], axis=-1)
    Jx *= position_scale
    Jy *= position_scale
    Jx[status != 1] = np.nan
    Jy[status != 1] = np.nan
    return Jx, Jy


def largest_singular_value(Jx, Jy):
    a = np.sum(Jx**2, axis=-1)
    b = np.sum(Jx * Jy, axis=-1)
    c = np.sum(Jy**2, axis=-1)
    eigenvalue = 0.5 * (a + c + np.sqrt((a - c)**2 + 4.0 * b**2))
    return np.sqrt(np.maximum(eigenvalue, 0.0))


def effective_t_end(data):
    names = [str(name) for name in data["sweep_names"]]
    if "tEnd" not in names:
        return float(data["t_end"])
    index = names.index("tEnd")
    return float(
        data["sweep_combinations"][
            combination_index,
            index,
        ]
    )


def parameter_text(data):
    names = [str(name) for name in data["sweep_names"]]
    values = data["sweep_combinations"][combination_index]
    if len(names) == 0:
        return "baseline"
    return ", ".join(
        f"{name} = {value:.6e}"
        for name, value
        in zip(names, values)
    )


def save_candidates(data, stretching_rate,):
    x = data["x"]
    y = data["y"]
    status = data["status"][combination_index]
    valid = (np.isfinite(stretching_rate) & (status == 1))
    flat_indices = np.flatnonzero(valid)
    order = flat_indices[
        np.argsort(
            stretching_rate.flat[
                flat_indices
            ]
        )[::-1]
    ]

    order = order[:number_of_candidates]

    lines = [
        (
            f"combination_index "
            f"{combination_index}"
        ),
        parameter_text(data),
        "",
    ]

    for rank, flat_index in enumerate(order, start=1):
        i, j = np.unravel_index(flat_index, stretching_rate.shape)
        lines.append(
            (
                f"{rank:02d}: "
                f"point_{flat_index:06d}, "
                f"x0 = {x[j]:.10e}, "
                f"y0 = {y[i]:.10e}, "
                f"stretching = "
                f"{stretching_rate[i, j]:.10e} 1/s, "
                f"reference = "
                f"trajectories/"
                f"combination_{combination_index:04d}/"
                f"point_{flat_index:06d}/input.in"
            )
        )

    with open(
        candidate_file,
        "w",
    ) as file:
        file.write(
            "\n".join(lines)
            + "\n"
        )

    print(
        "\n".join(lines)
    )


def main():
    data = np.load(input_file)
    Jx, Jy = projected_jacobian(data)
    sigma_max = largest_singular_value(Jx, Jy)
    t_end = effective_t_end(data)
    stretching_rate = (
        np.log(
            np.maximum(
                sigma_max,
                np.finfo(float).tiny,
            )
        )
        / t_end
    )
    save_candidates(data, stretching_rate)
    x = data["x"]
    y = data["y"]
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    image = ax.imshow(stretching_rate, origin="lower", extent=[x[0], x[-1], y[0], y[-1],], aspect="equal")
    ax.set_xlim([x[-1], x[0],])
    ax.set_ylim([y[-1], y[0],])
    ax.set_xlabel(r"$x_0 \, [\unit{m}]$")
    ax.set_ylabel(r"$y_0 \, [\unit{m}]$")
    ax.tick_params(direction="in", top=True, right=True)
    colorbar = fig.colorbar(image, ax=ax,)
    colorbar.set_label(r"$T^{-1}\log \sigma_{\max} \, [\unit{s^{-1}}]$")
    plt.savefig(output_file, dpi=300)
    

if __name__ == "__main__":
    main()
