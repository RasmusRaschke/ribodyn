from __future__ import annotations
from pathlib import Path
from manim import *
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
import numpy as np
import simutils as su


DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "rotatingEM.csv"
)


def quat_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)

    return Rotation.from_quat(
        [q[1], q[2], q[3], q[0]]
    ).as_matrix()


def safe_unit(vector, fallback=OUT):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return np.asarray(fallback, dtype=float)

    return vector / norm


class RotatingMagneticField(ThreeDScene):
    def construct(self):
        data = su.extract(DATA_FILE)[DATA_FILE.stem]
        t = np.asarray(data.t)

        def spline(values):
            return PchipInterpolator(t, np.asarray(values))

        qw = spline(data.qw)
        qx = spline(data.qx)
        qy = spline(data.qy)
        qz = spline(data.qz)

        mux = spline(data.mu_world_x)
        muy = spline(data.mu_world_y)
        muz = spline(data.mu_world_z)

        bx = spline(data.Bx_world)
        by = spline(data.By_world)
        bz = spline(data.Bz_world)

        total_energy = spline(data.E_total)

        time = ValueTracker(t[0])

        def body_rotation(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])

            return quat_matrix(
                qw(tt),
                qx(tt),
                qy(tt),
                qz(tt),
            )

        def magnetic_moment(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])

            return np.array(
                [mux(tt), muy(tt), muz(tt)]
            )

        def magnetic_field(sim_time):
            tt = np.clip(sim_time, t[0], t[-1])

            return np.array(
                [bx(tt), by(tt), bz(tt)]
            )

        self.set_camera_orientation(
            phi=65 * DEGREES,
            theta=-42 * DEGREES,
            zoom=0.9,
        )

        body = Sphere(
            radius=0.72,
            resolution=(40, 40),
            fill_opacity=0.75,
            stroke_width=1,
            color=GREY_BROWN,
        )

        def make_body_axes():
            rotation = body_rotation(
                time.get_value()
            )

            arrows = VGroup()

            for index, color in enumerate(
                (RED, GREEN, BLUE)
            ):
                arrows.add(
                    Arrow3D(
                        start=ORIGIN,
                        end=0.98 * rotation[:, index],
                        color=color,
                        thickness=0.018,
                        height=0.15,
                        base_radius=0.055,
                    )
                )

            return arrows

        body_axes = always_redraw(
            make_body_axes
        )

        def make_moment_arrow():
            direction = safe_unit(
                magnetic_moment(
                    time.get_value()
                ),
                RIGHT,
            )

            return Arrow3D(
                start=ORIGIN,
                end=1.65 * direction,
                color=RED_E,
                thickness=0.026,
                height=0.20,
                base_radius=0.075,
            )

        moment_arrow = always_redraw(
            make_moment_arrow
        )

        moment_trail = TracedPath(
            moment_arrow.get_end,
            stroke_color=RED_E,
            stroke_width=2,
            dissipating_time=3.0,
        )

        def make_field_lattice():
            direction = safe_unit(
                magnetic_field(
                    time.get_value()
                ),
                RIGHT,
            )

            arrows = VGroup()

            for gx in (-2.5, -1.25, 0.0, 1.25, 2.5):
                for gy in (-2.5, -1.25, 0.0, 1.25, 2.5):
                    start = np.array(
                        [gx, gy, -1.35]
                    )

                    arrows.add(
                        Arrow3D(
                            start=start,
                            end=start + 0.72 * direction,
                            color=BLUE_D,
                            thickness=0.009,
                            height=0.10,
                            base_radius=0.035,
                        )
                    )

            return arrows

        field_lattice = always_redraw(
            make_field_lattice
        )

        def make_field_reference_arrow():
            direction = safe_unit(
                magnetic_field(
                    time.get_value()
                ),
                RIGHT,
            )

            return Arrow3D(
                start=ORIGIN,
                end=2.15 * direction,
                color=BLUE_D,
                thickness=0.023,
                height=0.18,
                base_radius=0.07,
            )

        field_arrow = always_redraw(
            make_field_reference_arrow
        )

        title = Text(
            "Rigid dipole in a rotating magnetic field",
            font_size=30,
        ).to_corner(UL)

        equation = MathTex(
            r"\mathbf B(t)=R_{\hat{\mathbf z}}(\omega_f t)\mathbf B_0,"
            r"\qquad"
            r"\boldsymbol\tau_b=\boldsymbol\mu_b\times\mathbf B_b",
            font_size=27,
        ).next_to(
            title,
            DOWN,
            aligned_edge=LEFT,
        )

        energy_label = Text(
            "Mechanical energy:",
            font_size=22,
        ).to_corner(DL)

        energy_value = DecimalNumber(
            total_energy(t[0]),
            num_decimal_places=5,
            font_size=22,
        ).next_to(
            energy_label,
            RIGHT,
        )

        energy_value.add_updater(
            lambda mob: mob.set_value(
                total_energy(
                    time.get_value()
                )
            )
        )

        note = Text(
            "Not conserved: the prescribed field performs work",
            font_size=18,
        ).next_to(
            energy_label,
            UP,
            aligned_edge=LEFT,
        )

        self.add_fixed_in_frame_mobjects(
            title,
            equation,
            note,
            energy_label,
            energy_value,
        )

        self.add(
            field_lattice,
            moment_trail,
            body,
            body_axes,
            field_arrow,
            moment_arrow,
        )

        self.play(
            time.animate.set_value(t[-1]),
            run_time=15,
            rate_func=linear,
        )

        self.wait(0.5)
