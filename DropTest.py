"""
Cylinder drop-angle simulation
================================
Models a steel pipe released from a ramp at a chosen angle, sliding
nose-first down a slightly rough incline, entering a water tank nose-first,
and sinking under gravity, buoyancy, quadratic drag, and hydrodynamic torque
until it settles on the tank floor.

Pipe modelled : STZ 3/8" × 3" Schedule 40 galvanised steel nipple
                OD = 0.675 in, wall = 0.091 in, ID = 0.493 in, length = 3 in
                Open-ended: buoyancy acts on steel cross-section only.

Independent variable : ramp angle (45 deg, 60 deg, 75 deg by default)
Dependent variable    : horizontal displacement underwater (cm)

Physics:
- Nose-first entry: cylinder axis aligned with ramp direction at water entry
- Hollow pipe: mass and buoyancy use steel annular volume (open ends)
- Decomposed drag: axial CD = 0.82, broadside CD = 1.2, applied per component
- Orientation-dependent added mass: interpolates MA_AXIAL → MA_NORMAL
- Munk moment: potential-flow torque that drives the cylinder toward broadside
- Flow-speed-scaled pitch damping: prevents continuous tumbling
- Partial-submersion ramp-up: forces grow over the orientation-dependent entry length
- Reynolds-number-dependent drag coefficient (empirical fit)

Run:  python DropTest.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ── Physical constants ────────────────────────────────────────────────────────
G         = 9.81          # m/s²
RHO_WATER = 1000.0        # kg/m³
MU_WATER  = 0.001         # Pa·s, dynamic viscosity at ~20 °C

# Drag coefficients for a finite cylinder
CD_BROADSIDE = 1.2        # normal to axis (crossflow)
CD_AXIAL     = 0.82       # along axis (fore-aft)

# Cylinder geometry — STZ 3/8" × 3" Schedule 40 galvanised nipple
CYL_RADIUS_OUTER = 0.675 / 2 * 0.0254   # 0.008573 m  (OD = 0.675")
CYL_RADIUS_INNER = 0.493 / 2 * 0.0254   # 0.006261 m  (ID = 0.493")
CYL_RADIUS       = CYL_RADIUS_OUTER      # outer radius used for drag and geometry
CYL_LENGTH       = 3.0 * 0.0254         # 0.0762 m  (3")
# Open-ended pipe: buoyancy acts on steel annulus only, not full outer volume
CYL_VOLUME_STEEL = np.pi * (CYL_RADIUS_OUTER**2 - CYL_RADIUS_INNER**2) * CYL_LENGTH
CYL_VOLUME       = CYL_VOLUME_STEEL
CYL_AREA_BROADSIDE = CYL_LENGTH * 2 * CYL_RADIUS_OUTER
CYL_AREA_AXIAL     = np.pi * CYL_RADIUS_OUTER**2

# Steel
RHO_STEEL = 7800.0
CYL_MASS  = RHO_STEEL * CYL_VOLUME_STEEL
# Hollow-cylinder MOI about transverse centre axis: (1/12)*m*(3(r_o²+r_i²) + L²)
CYL_MOI   = (1.0 / 12.0) * CYL_MASS * (3.0 * (CYL_RADIUS_OUTER**2 + CYL_RADIUS_INNER**2) + CYL_LENGTH**2)

# Added masses — based on outer geometry (fluid sees outer surface)
MA_NORMAL = RHO_WATER * np.pi * CYL_RADIUS_OUTER**2 * CYL_LENGTH  # broadside added mass
MA_AXIAL  = RHO_WATER * (8.0 / 3.0) * CYL_RADIUS_OUTER**3         # axial added mass (Lamb disk)

# Ramp / tank defaults
TANK_DEPTH      = 0.205  # m — measured water depth
RELEASE_HEIGHT  = 0.10   # m from water surface to pipe centre of mass at release
DEFAULT_MU      = 0.20   # kinetic friction — wet steel on acrylic (PMMA)

ZETA_PITCH  = 1.6   # pitch damping ratio — overdamped prevents oscillation past broadside

PLOT_COLORS = {30: "#2F6FB8", 45: "#C77F1B", 60: "#B5384B"}


# ── Drag coefficient vs Reynolds number ───────────────────────────────────────
def cd_vs_re(Re):
    """Empirical Cd for a cylinder in crossflow."""
    if Re < 1.0:
        return 24.0 / max(Re, 1e-9)
    if Re < 1e3:
        return 24.0 / Re + 6.0 / (1.0 + np.sqrt(Re)) + 0.4
    if Re < 2e5:
        return 1.0
    return 0.3   # supercritical drag crisis


# ── Ramp phase ────────────────────────────────────────────────────────────────
def ramp_exit_velocity(theta_deg, release_height=RELEASE_HEIGHT, mu=DEFAULT_MU):
    """Exit speed (m/s) and travel time (s) given fixed release height above water."""
    theta = np.radians(theta_deg)
    a = G * (np.sin(theta) - mu * np.cos(theta))
    if a <= 0:
        min_ang = np.degrees(np.arctan(mu))
        raise ValueError(
            f"Ramp angle {theta_deg}° too shallow for mu={mu:.2f}. "
            f"Minimum sliding angle: {min_ang:.1f}°"
        )
    # Centre is 10cm above water; nose contacts surface when centre has dropped
    # h - (L/2)*sin(theta), so the ramp travel is shorter than h/sin(theta)
    h_to_entry = release_height - (CYL_LENGTH / 2.0) * np.sin(theta)
    ramp_length = h_to_entry / np.sin(theta)
    v_exit = np.sqrt(2.0 * a * ramp_length)
    return v_exit, v_exit / a, ramp_length


# ── Underwater ODE ────────────────────────────────────────────────────────────
def underwater_derivatives(t, state, mass, moi):
    """
    State: [x, y, vx, vy, theta, omega]
      x, y    : position (y = depth, positive downward)
      vx, vy  : velocity
      theta   : cylinder axis angle from horizontal (rad)
                theta = ramp_angle at entry (nose-first along ramp)
                theta → 0 at broadside-to-downward-flow equilibrium
      omega   : angular velocity (rad/s)
    """
    x, y, vx, vy, theta, omega = state
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    speed = np.hypot(vx, vy)

    # ── Body-frame velocity decomposition ────────────────────────────────────
    # ax_hat = (cos_t, sin_t): nose direction in world frame
    v_ax =  vx * cos_t + vy * sin_t   # along axis (positive = nose direction)
    v_n  = -vx * sin_t + vy * cos_t   # normal to axis

    # ── Partial-submersion fraction ───────────────────────────────────────────
    # Entry length = vertical projection of cylinder at current orientation
    entry_len = CYL_LENGTH * abs(sin_t) + 2.0 * CYL_RADIUS * abs(cos_t)
    f_sub = float(np.clip(y / max(entry_len, 1e-9), 0.0, 1.0))

    # ── Orientation-dependent added mass ──────────────────────────────────────
    # At theta=0 (broadside): m_added = MA_NORMAL (max, resists normal motion)
    # At theta=90° (axial):   m_added = MA_AXIAL  (min, resists axial motion)
    m_added = MA_NORMAL * cos_t**2 + MA_AXIAL * sin_t**2
    m_eff   = mass + m_added * f_sub

    # ── Reynolds-number-dependent drag coefficients ───────────────────────────
    D  = 2.0 * CYL_RADIUS
    Re = RHO_WATER * speed * D / MU_WATER if speed > 1e-9 else 1.0
    cd_n  = cd_vs_re(Re) * CD_BROADSIDE
    cd_ax = cd_vs_re(Re) * CD_AXIAL

    # ── Drag in body frame (quadratic, opposing motion) ───────────────────────
    k_ax = 0.5 * RHO_WATER * cd_ax * CYL_AREA_AXIAL     / m_eff
    k_n  = 0.5 * RHO_WATER * cd_n  * CYL_AREA_BROADSIDE / m_eff

    a_ax = -k_ax * abs(v_ax) * v_ax * f_sub
    a_n  = -k_n  * abs(v_n)  * v_n  * f_sub

    # Rotate body-frame drag to world frame
    ax_drag = a_ax * cos_t - a_n * sin_t
    ay_drag = a_ax * sin_t + a_n * cos_t

    # ── Gravity and buoyancy ──────────────────────────────────────────────────
    ay_body = (mass * G - RHO_WATER * CYL_VOLUME * G * f_sub) / m_eff

    ax_total = ax_drag
    ay_total = ay_drag + ay_body

    # ── Rotational dynamics ───────────────────────────────────────────────────
    # Munk moment: drives cylinder toward broadside (theta → 0) for downward flow.
    # tau = -(MA_NORMAL - MA_AXIAL) * v_axial * v_normal
    #   zero at entry (v_n = 0) and at equilibrium (v_ax = 0 when theta = 0)
    tau_munk = -(MA_NORMAL - MA_AXIAL) * v_ax * v_n

    # Flow-speed-scaled pitch damping, referenced to critical damping
    K_spring = max((MA_NORMAL - MA_AXIAL) * speed**2, 1e-9)
    c_crit   = 2.0 * np.sqrt(moi * K_spring)
    tau_damp = -ZETA_PITCH * c_crit * omega

    # Scale all fluid torque by submersion fraction
    domega = (tau_munk + tau_damp) * f_sub / moi

    return [vx, vy, ax_total, ay_total, omega, domega]


def _make_bottom_event(max_depth):
    def hit_bottom(t, state, *_):
        return max_depth - state[1]
    hit_bottom.terminal  = True
    hit_bottom.direction = -1
    return hit_bottom


# ── Full single-trial simulation ──────────────────────────────────────────────
def simulate_drop(theta_deg, release_height=RELEASE_HEIGHT,
                  mu=DEFAULT_MU, tank_depth=TANK_DEPTH):
    """Simulate one drop. Returns a dict with trajectory and summary results."""
    v_exit, t_ramp, ramp_length = ramp_exit_velocity(theta_deg, release_height, mu)
    theta_rad = np.radians(theta_deg)
    vx0 = v_exit * np.cos(theta_rad)
    vy0 = v_exit * np.sin(theta_rad)

    # Ramp trajectory (30-point closed form)
    a_ramp    = G * (np.sin(theta_rad) - mu * np.cos(theta_rad))
    ramp_t    = np.linspace(0, t_ramp, 30)
    remaining = ramp_length - 0.5 * a_ramp * ramp_t**2
    ramp_x    = -remaining * np.cos(theta_rad)
    ramp_y    = -remaining * np.sin(theta_rad)

    # Initial orientation: nose aligned with ramp (nose pointing down-slope)
    # theta = theta_rad > 0: nose pointing rightward and downward ✓
    theta0 = theta_rad
    omega0 = 0.0

    sol = solve_ivp(
        underwater_derivatives,
        t_span=(0, 10.0),
        y0=[0.0, 0.0, vx0, vy0, theta0, omega0],
        args=(CYL_MASS, CYL_MOI),
        events=_make_bottom_event(tank_depth - CYL_RADIUS),
        max_step=0.002,
        rtol=1e-6, atol=1e-9,
    )

    return {
        "angle_deg":             theta_deg,
        "release_height_cm":     release_height * 100,
        "friction_coeff":        mu,
        "entry_speed_cm_s":      v_exit * 100,
        "ramp_time_s":           t_ramp,
        "settle_time_s":         sol.t[-1],
        "displacement_cm":       sol.y[0, -1] * 100,
        "final_orientation_deg": np.degrees(sol.y[4, -1]),
        "ramp_traj_cm":          (ramp_x * 100, ramp_y * 100),
        "water_traj_cm":         (sol.y[0] * 100, sol.y[1] * 100),
        "water_theta_deg":       np.degrees(sol.y[4]),
        "water_t":               sol.t,
        "final_vx":              sol.y[2, -1],
        "final_vy":              sol.y[3, -1],
    }


# ── Batch run ─────────────────────────────────────────────────────────────────
def run_experiment(angles=(45, 60, 75), release_height=RELEASE_HEIGHT,
                   mu=DEFAULT_MU):
    results = [simulate_drop(a, release_height, mu) for a in angles]
    df = pd.DataFrame([{
        "angle_deg":             r["angle_deg"],
        "entry_speed_cm_s":      round(r["entry_speed_cm_s"], 1),
        "ramp_time_s":           round(r["ramp_time_s"], 3),
        "settle_time_s":         round(r["settle_time_s"], 3),
        "displacement_cm":       round(r["displacement_cm"], 2),
        "final_orientation_deg": round(r["final_orientation_deg"], 1),
    } for r in results])
    return df, results


# ── Self-test ─────────────────────────────────────────────────────────────────
def self_test():
    """Validate physics against analytical benchmarks."""
    print("Running self-test...")
    passed = True

    net_weight = (CYL_MASS - RHO_WATER * CYL_VOLUME) * G

    # Analytic broadside terminal speed (at constant v, added mass irrelevant)
    v_t_broad = np.sqrt(net_weight / (0.5 * RHO_WATER * CD_BROADSIDE * CYL_AREA_BROADSIDE))

    # 1. Terminal velocity: drop at 45° into deep tank, check final speed
    r = simulate_drop(45, tank_depth=3.0)
    v_final = np.hypot(r["final_vx"], r["final_vy"])
    err_pct = abs(v_final - v_t_broad) / v_t_broad * 100
    ok = err_pct < 15.0
    print(f"  [{'PASS' if ok else 'FAIL'}] Terminal speed: "
          f"{v_final * 100:.1f} cm/s  analytic {v_t_broad * 100:.1f} cm/s  "
          f"({err_pct:.1f}% error)")
    passed = passed and ok

    # 2. Broadside settling: final theta within ±20° of 0
    final_theta = r["final_orientation_deg"]
    ok = abs(final_theta) < 20.0
    print(f"  [{'PASS' if ok else 'FAIL'}] Final orientation: "
          f"{final_theta:.1f}°  (broadside = 0°, tol ±20°)")
    passed = passed and ok

    # 3. Monotonic displacement: steeper angle → less horizontal travel
    df, _ = run_experiment(angles=(30, 45, 60))
    d = df["displacement_cm"].tolist()
    ok = d[0] > d[1] > d[2]
    print(f"  [{'PASS' if ok else 'FAIL'}] Displacement monotonic: "
          f"{d[0]:.2f} > {d[1]:.2f} > {d[2]:.2f} cm")
    passed = passed and ok

    print(f"Self-test {'PASSED' if passed else 'FAILED'}.")
    return passed


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_trajectories(results, filename="trajectories.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        c = PLOT_COLORS.get(r["angle_deg"], "gray")
        rx, ry = r["ramp_traj_cm"]
        wx, wy = r["water_traj_cm"]
        ax.plot(rx, ry, "--", color=c, alpha=0.5)
        ax.plot(wx, wy, "-", color=c, linewidth=2, label=f"{r['angle_deg']}°")
        ax.plot(wx[-1], wy[-1], "o", color=c)
    ax.axhline(0, color="steelblue", linestyle=":", label="water surface")
    ax.set_xlabel("Horizontal distance from entry point (cm)")
    ax.set_ylabel("Depth (cm)")
    ax.invert_yaxis()
    ax.set_title("Steel cylinder trajectory by drop angle (nose-first entry)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return fig


def plot_displacement_vs_angle(df, filename="displacement_by_angle.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [PLOT_COLORS.get(a, "gray") for a in df["angle_deg"]]
    ax.bar(df["angle_deg"].astype(str) + "°", df["displacement_cm"], color=colors)
    ax.set_xlabel("Drop angle")
    ax.set_ylabel("Horizontal displacement (cm)")
    ax.set_title("Horizontal displacement vs. drop angle (Steel, nose-first)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return fig


def plot_orientation_history(results, filename="orientation_history.png"):
    """Shows how the cylinder pitch angle evolves as it sinks toward broadside."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        c = PLOT_COLORS.get(r["angle_deg"], "gray")
        ax.plot(r["water_t"] * 1000, r["water_theta_deg"],
                color=c, linewidth=2, label=f"{r['angle_deg']}°")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.6,
               label="broadside (θ = 0°, axis horizontal)")
    ax.set_xlabel("Time underwater (ms)")
    ax.set_ylabel("Cylinder orientation θ (° from horizontal)")
    ax.set_title("Nose-first entry → broadside settling")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return fig


def plot_angle_sweep(angle_range=np.arange(35, 86, 2),
                     release_height=RELEASE_HEIGHT, mu=DEFAULT_MU,
                     filename="angle_sweep.png"):
    disps = [simulate_drop(a, release_height, mu)["displacement_cm"]
             for a in angle_range]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(angle_range, disps, color="#534AB7", linewidth=2)
    for a in (30, 45, 60):
        ax.axvline(a, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Drop angle (degrees)")
    ax.set_ylabel("Horizontal displacement (cm)")
    ax.set_title(f"Predicted displacement vs. angle (Steel, μ={DEFAULT_MU})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return fig, angle_range, disps


def animate_trajectories(results, filename="trajectories_animated.gif",
                          n_frames=120, interval=50):
    """Animated GIF showing all trajectories simultaneously."""
    angle_data = []
    for r in results:
        rx, ry = r["ramp_traj_cm"]
        wx, wy = r["water_traj_cm"]
        full_x   = np.concatenate([rx, wx[1:]])
        full_y   = np.concatenate([ry, wy[1:]])
        ramp_end = len(rx) - 1

        ds     = np.hypot(np.diff(full_x), np.diff(full_y))
        s      = np.concatenate([[0], np.cumsum(ds)])
        s_norm = s / s[-1]
        t_uni  = np.linspace(0, 1, n_frames)

        anim_x = np.interp(t_uni, s_norm, full_x)
        anim_y = np.interp(t_uni, s_norm, full_y)
        entry_frame = int(np.searchsorted(t_uni, s[ramp_end] / s[-1]))

        angle_data.append({
            "color":       PLOT_COLORS.get(r["angle_deg"], "gray"),
            "label":       f"{r['angle_deg']}°",
            "anim_x":      anim_x,
            "anim_y":      anim_y,
            "entry_frame": entry_frame,
        })

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="steelblue", linestyle=":", label="water surface")
    ax.set_xlabel("Horizontal distance from entry point (cm)")
    ax.set_ylabel("Depth (cm)")
    ax.set_title("Cylinder trajectory by drop angle")
    ax.grid(alpha=0.3)

    all_x = np.concatenate([d["anim_x"] for d in angle_data])
    all_y = np.concatenate([d["anim_y"] for d in angle_data])
    px = max(1.0, (all_x.max() - all_x.min()) * 0.05)
    py = max(1.0, (all_y.max() - all_y.min()) * 0.05)
    ax.set_xlim(all_x.min() - px, all_x.max() + px)
    ax.set_ylim(all_y.max() + py, all_y.min() - py)

    for d in angle_data:
        c = d["color"]
        d["ramp_line"],  = ax.plot([], [], "--", color=c, alpha=0.5)
        d["water_line"], = ax.plot([], [], "-",  color=c, linewidth=2, label=d["label"])
        d["dot"],        = ax.plot([], [], "o",  color=c, markersize=8)
    ax.legend()

    def init():
        artists = []
        for d in angle_data:
            d["ramp_line"].set_data([], [])
            d["water_line"].set_data([], [])
            d["dot"].set_data([], [])
            artists += [d["ramp_line"], d["water_line"], d["dot"]]
        return artists

    def update(frame):
        artists = []
        for d in angle_data:
            ef = d["entry_frame"]
            xd, yd = d["anim_x"], d["anim_y"]
            if frame <= ef:
                d["ramp_line"].set_data(xd[:frame + 1], yd[:frame + 1])
                d["water_line"].set_data([], [])
            else:
                d["ramp_line"].set_data(xd[:ef + 1], yd[:ef + 1])
                d["water_line"].set_data(xd[ef:frame + 1], yd[ef:frame + 1])
            d["dot"].set_data([xd[frame]], [yd[frame]])
            artists += [d["ramp_line"], d["water_line"], d["dot"]]
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=interval, blit=True)
    anim.save(filename, writer="pillow", fps=1000 // interval)
    plt.close(fig)
    return anim


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    self_test()
    print()

    df, results = run_experiment(angles=(30, 45, 60))
    print(df.to_string(index=False))

    plot_trajectories(results)
    plot_displacement_vs_angle(df)
    plot_orientation_history(results)
    plot_angle_sweep()
    animate_trajectories(results)

    print("\nSaved: trajectories.png, displacement_by_angle.png, "
          "orientation_history.png, angle_sweep.png, trajectories_animated.gif")
