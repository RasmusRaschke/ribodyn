import csv
import subprocess
import tempfile
from multiprocessing import Pool
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
base_directory = Path(__file__).resolve().parent
solver = (
    base_directory
    / "../../build/solver"
).resolve()
reference_input = (
    base_directory
    / "trajectories"
    / "combination_0000"
    / "point_000000"
    / "input.in"
)
output_directory = (
    base_directory
    / "chaos_full"
)
cores = 40
segment_time = 0.02
segments = 250
epsilon = 1e-6
# Make tangent vectors dimensionless by appropriate scaling
position_scale = 0.01
angle_scale = 1.0
velocity_scale = 0.1
omega_scale = 100.0
initial_tangent = np.ones(12)
##############################################################################

dof_names = [
    "r_x",
    "r_y",
    "r_z",
    "theta_x",
    "theta_y",
    "theta_z",
    "v_x",
    "v_y",
    "v_z",
    "Omega_x",
    "Omega_y",
    "Omega_z",
]

dof_labels = [
    r"$r_x$",
    r"$r_y$",
    r"$r_z$",
    r"$\theta_x$",
    r"$\theta_y$",
    r"$\theta_z$",
    r"$v_x$",
    r"$v_y$",
    r"$v_z$",
    r"$\Omega_x$",
    r"$\Omega_y$",
    r"$\Omega_z$",
]

scales = np.array([position_scale] * 3 + [angle_scale] * 3 + [velocity_scale] * 3 + [omega_scale] * 3)

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2]-q[3]])


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quaternion_exponential(rotation_vector):
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-14:
        q = np.concatenate([np.array([1.0]), 0.5 * rotation_vector])
        return normalize_quaternion(q)
    axis = (rotation_vector / angle)
    return np.concatenate([np.array([np.cos(0.5 * angle)]), np.sin(0.5 * angle) * axis])


def relative_rotation_vector(q_ref, q_other):
    q_ref = normalize_quaternion(q_ref)
    q_other = normalize_quaternion(q_other)
    q_rel = quaternion_multiply(quaternion_conjugate(q_ref),q_other)
    q_rel = normalize_quaternion(q_rel)
    if q_rel[0] < 0.0:
        q_rel = -q_rel
    vector = q_rel[1:]
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-14:
        return 2.0 * vector
    angle = 2.0 * np.arctan2(vector_norm, q_rel[0])
    return (angle * vector / vector_norm)


def read_input_state(filename):
    values = {}

    with open(
        filename,
        "r",
    ) as file:
        for line in file:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            values[parts[0]] = parts[1:]
    r = np.array(values["position"], dtype=float)
    v = np.array(values["velocity"], dtype=float)
    Omega = np.array(values["omega"], dtype=float)
    q = np.array(values["quaternion"], dtype=float)
    return np.concatenate([r, q, v, Omega])


def state_difference(reference, perturbed):
    delta_r = (perturbed[0:3]- reference[0:3])
    delta_theta = (relative_rotation_vector(reference[3:7], perturbed[3:7]))
    delta_v = (perturbed[7:10] - reference[7:10])
    delta_Omega = (perturbed[10:13] - reference[10:13])
    return np.concatenate([delta_r, delta_theta, delta_v, delta_Omega])


def normalized_difference(reference, perturbed,):
    return (state_difference(reference,  perturbed) / scales)


def perturb_state(reference, tangent_direction,):
    physical_perturbation = (epsilon * scales * tangent_direction)
    perturbed = reference.copy()
    perturbed[0:3] += (physical_perturbation[0:3])
    delta_q = quaternion_exponential(physical_perturbation[3:6])
    perturbed[3:7] = (quaternion_multiply(reference[3:7], delta_q))
    perturbed[3:7] = (normalize_quaternion(perturbed[3:7]))
    perturbed[7:10] += (physical_perturbation[6:9])
    perturbed[10:13] += (physical_perturbation[9:12])
    return perturbed


def make_input(base_input, state, current_t_end):
    lines = [
        "",
        "# chaos_full.py state override",
        (
            "position "
            f"{state[0]:.17g} "
            f"{state[1]:.17g} "
            f"{state[2]:.17g}"
        ),
        (
            "quaternion "
            f"{state[3]:.17g} "
            f"{state[4]:.17g} "
            f"{state[5]:.17g} "
            f"{state[6]:.17g}"
        ),
        (
            "velocity "
            f"{state[7]:.17g} "
            f"{state[8]:.17g} "
            f"{state[9]:.17g}"
        ),
        (
            "omega "
            f"{state[10]:.17g} "
            f"{state[11]:.17g} "
            f"{state[12]:.17g}"
        ),
        f"tEnd {current_t_end:.17g}",
        "",
    ]
    return (base_input.rstrip() + "\n" + "\n".join(lines))


def read_final_state(filename):
    with open(filename, "r", newline="",) as file:
        reader = csv.DictReader(file)
        final_row = None
        for row in reader:
            final_row = row
    if final_row is None:
        raise RuntimeError(
            "The solver produced no output rows."
        )
    return np.array([
        float(final_row["x"]),
        float(final_row["y"]),
        float(final_row["z"]),
        float(final_row["qw"]),
        float(final_row["qx"]),
        float(final_row["qy"]),
        float(final_row["qz"]),
        float(final_row["vx"]),
        float(final_row["vy"]),
        float(final_row["vz"]),
        float(final_row["Ox"]),
        float(final_row["Oy"]),
        float(final_row["Oz"]),
    ])


def run_state(task):
    (base_input, state, current_t_end) = task
    with tempfile.TemporaryDirectory(prefix="ribodyn_chaos_") as workdir:
        workdir = Path(workdir)
        input_name = (workdir / "input.in")
        output_name = (workdir / "output.csv")
        with open(input_name, "w",) as file:
            file.write(make_input(base_input, state, current_t_end))
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
                "Solver failed.\n\n"
                + process.stderr
            )
        return read_final_state(output_name)


def make_local_jacobian(reference_start, reference_end, perturbed_end):
    jacobian = np.empty((12, 12))
    for j in range(12):
        jacobian[:, j] = (normalized_difference(reference_end, perturbed_end[j]) / epsilon)
    return jacobian


def save_reference_trajectory(times, states):
    filename = (output_directory / "reference.csv")
    header = (
        "t,"
        "x,y,z,"
        "qw,qx,qy,qz,"
        "vx,vy,vz,"
        "Ox,Oy,Oz"
    )
    output = np.column_stack([times,states])
    np.savetxt(filename, output, delimiter=",", header=header, comments="")


def plot_largest_lyapunov(times, largest_lyapunov):
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.plot(times, largest_lyapunov, linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel(r"$t \, [\unit{s}]$")
    ax.set_ylabel(r"$\lambda_1(t) \, [\unit{s^{-1}}]$")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, alpha=0.3)
    plt.savefig(output_directory / "largest_lyapunov.pdf", dpi=300,)


def plot_jacobian(log10_abs_jacobian,):
    fig, ax = plt.subplots(
        figsize=(11, 9),
        layout="constrained",
    )
    image = ax.imshow(
        log10_abs_jacobian,
        origin="upper",
        aspect="equal",
    )
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(12))
    ax.set_xticklabels(dof_labels, rotation=45, ha="right")
    ax.set_yticklabels(dof_labels)
    ax.set_xlabel(r"$\textrm{initial tangent coordinate}$")
    ax.set_ylabel(r"$\textrm{final tangent coordinate}$")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}|\bar J_{ij}|$")
    plt.savefig(output_directory / "jacobian_scaling.pdf", dpi=300)


def main():
    if not solver.is_file():
        raise FileNotFoundError(
            f"Solver not found:\n{solver}"
        )
    if not reference_input.is_file():
        raise FileNotFoundError(
            f"Reference input not found:\n"
            f"{reference_input}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    base_input = (reference_input.read_text())
    reference = read_input_state(reference_input)
    reference = run_state((base_input, reference, 0.0))
    tangent = np.asarray(initial_tangent, dtype=float)
    tangent /= np.linalg.norm(tangent)
    accumulated_log_growth = 0.0
    times = np.arange(segments + 1, dtype=float) * segment_time
    reference_states = np.empty((segments + 1, 13))
    reference_states[0] = (reference)
    largest_lyapunov = np.full(segments, np.nan)
    local_stretching = np.full(segments, np.nan)
    numerical_rank = np.zeros(segments, dtype=int)
    cumulative_jacobian = np.eye(12)
    cumulative_log_scale = 0.0
    with Pool(processes=cores) as pool:
        for segment in range(segments):
            perturbations = [perturb_state(reference, np.eye(12)[j],) for j in range(12)]
            tasks = [(base_input, reference, segment_time)] + [(base_input, perturbed, segment_time) for perturbed in perturbations]
            results = pool.map(run_state, tasks)
            reference_end = (results[0])
            perturbed_end = (results[1:])
            jacobian = (make_local_jacobian(reference, reference_end, perturbed_end))
            singular_values = (np.linalg.svd(jacobian, compute_uv=False))
            numerical_rank[segment] = np.sum(singular_values > (1e-9 * singular_values[0]))
            local_stretching[segment] = (np.log(max(singular_values[0], np.finfo(float).tiny,)) / segment_time)
            tangent_next = (jacobian @ tangent)
            growth = np.linalg.norm(tangent_next)
            accumulated_log_growth += (np.log(max(growth, np.finfo(float).tiny)))
            tangent = (tangent_next / growth)
            largest_lyapunov[segment] = (accumulated_log_growth / ((segment + 1) * segment_time))
            cumulative_jacobian = (jacobian @ cumulative_jacobian)
            cumulative_norm = (np.linalg.norm(cumulative_jacobian, ord="fro"))
            if (cumulative_norm > 1e100 or (cumulative_norm > 0.0 and cumulative_norm < 1e-100)):
                cumulative_jacobian /= (cumulative_norm)
                cumulative_log_scale += (np.log(cumulative_norm))
            reference = (reference_end)
            reference_states[segment + 1] = reference
            print(
                f"Segment {segment + 1} / {segments}, "
                f"lambda_1 = "
                f"{largest_lyapunov[segment]:.6e} 1/s, "
                f"rank = "
                f"{numerical_rank[segment]}",
                flush=True,
            )

    with np.errstate(divide="ignore", invalid="ignore",):
        log10_abs_jacobian = (np.log10(np.abs(cumulative_jacobian)) + cumulative_log_scale / np.log(10.0))
    np.savez(
        output_directory
        / "chaos_full.npz",
        time=times[1:],
        largest_lyapunov=largest_lyapunov,
        local_stretching=local_stretching,
        numerical_rank=numerical_rank,
        log10_abs_jacobian=log10_abs_jacobian,
        state_scales=scales,
        epsilon=epsilon,
        segment_time=segment_time,
        segments=segments,
    )
    save_reference_trajectory(times, reference_states)
    plot_largest_lyapunov(times[1:], largest_lyapunov)
    plot_jacobian(log10_abs_jacobian,)
    print(
        "\nLargest Lyapunov exponent = "
        f"{largest_lyapunov[-1]:.8e} 1/s"
    )
    print(
        "Median numerical tangent rank = "
        f"{np.median(numerical_rank):.1f}"
    )
    print(
        "\nNote: segment restarts reset the solver time to zero. "
        "This script is therefore intended for autonomous systems. "
        "For explicitly time-dependent fields, the solver needs an "
        "initial-time input before this calculation is valid."
    )


if __name__ == "__main__":
    main()
