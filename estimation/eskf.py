"""15-state error-state Kalman filter for strapdown inertial navigation.

State: position, velocity, attitude error, accelerometer bias, gyro bias -
three each. The nominal state is propagated by ordinary strapdown mechanisation
and the filter tracks only the *error* in it, which keeps the attitude in a
quaternion (no gimbal trouble, no small-angle assumption on the nominal state)
while the covariance stays linear in a 3-vector of tilt errors.

The whole noise model is derived from the bench characterisation rather than
hand-tuned: white noise from the Allan deviation (ARW/VRW), bias random walks
from the bias-instability floor and the tau it sits at, and initial
uncertainties from the calibration's own stated ambiguities. See
sensor_config.py.

Three measurements, all of them available only while the skater is standing
still, which is why the stationary detector matters so much:

  ZUPT   velocity is zero               - removes accumulated velocity error,
                                          and through the covariance the
                                          accelerometer bias behind it
  ZARU   the measured rate *is* the bias - the only thing that observes gyro
                                          bias about the vertical, and so the
                                          only thing that bounds heading drift
  tilt   the measured specific force
         points along gravity            - observes roll and pitch

Nothing observes heading in an *absolute* sense: a 6-axis IMU has no compass, so
yaw starts at 0 by definition and is relative to the capture's start. The
initial yaw covariance says so honestly (180 degrees).

Heading is nevertheless partly observable *relative to the path*, and that
surprises people reading the output: after a stretch of motion, the first ZUPT
can rotate the frame by several degrees in one step. That is not drift. While
the skater moves, heading error and velocity error become correlated through
the covariance, so zeroing the velocity also corrects the heading that produced
it - this is how ZUPT-aided pedestrian inertial navigation recovers heading at
all. You can watch it happen: the yaw sigma drops sharply at the same sample.
Suppressing it (by shrinking p0_yaw, or by projecting the injected error onto
the tilt subspace) would give a smoother-looking heading and a worse path.

Frame: world z is UP, gravity is (0, 0, -g0), and a positive yaw is a left turn
seen from above. The specific force a level sensor reports is (0, 0, +g0) in
world coordinates, so world linear acceleration is ``R @ (a - ba) + g_world``.

Note that analysis/pc_visualizer*.py in this repository's history used the
mirror-image convention (world z down), which puts a top-down XY plot from
those modules in the opposite handedness. Do not mix results from the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from estimation.sensor_config import SensorConfig

G0 = 9.80665
DEG = math.pi / 180.0

# A gap longer than this is a hole in the capture, not a sample interval:
# propagating across it would manufacture motion out of nothing.
MAX_DT_S = 0.5


# --------------------------------------------------------------- quaternions
# Hamilton convention, [w, x, y, z], rotating body vectors into world.


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    """Exact, and safe at zero (where the axis is undefined)."""
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]])
    axis = v / theta
    s = math.sin(0.5 * theta)
    return np.array([math.cos(0.5 * theta), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Z-Y-X, matching matrix_to_euler()."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def matrix_to_euler(R: np.ndarray):
    """(roll, pitch, yaw) in radians, Z-Y-X."""
    pitch = -math.asin(float(np.clip(R[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1e-6:
        return 0.0, pitch, math.atan2(-R[0, 1], R[1, 1])
    return (math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0]))


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def accel_to_roll_pitch(accel_ms2: np.ndarray):
    """Levelling from a gravity vector. Yaw is unobservable and stays 0."""
    norm = float(np.linalg.norm(accel_ms2))
    if norm < 1e-6:
        return 0.0, 0.0
    ax, ay, _ = accel_ms2 / norm
    pitch = math.asin(float(np.clip(ax, -1.0, 1.0)))
    cp = max(math.cos(pitch), 1e-6)
    roll = math.asin(float(np.clip(-ay / cp, -1.0, 1.0)))
    return roll, pitch


# -------------------------------------------------------------- noise model


def _gyro_bias_repeatability(raw: dict, default: float = 0.022) -> float:
    """The run-to-run gyro bias spread the config records among its caveats."""
    for line in raw.get("bias", {}).get("caveats", []):
        if "gyro_bias_repeatability_dps" in line:
            for token in line.replace(":", " ").split():
                try:
                    return float(token)
                except ValueError:
                    continue
    return default


def hf_sigma(x: np.ndarray) -> np.ndarray:
    """White-noise sigma from the second difference (any smooth signal cancels)."""
    d = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return np.std(d, axis=0) / math.sqrt(6.0)


def capture_noise_ratio(accel_g: np.ndarray, gyro_dps: np.ndarray,
                        stationary: np.ndarray, cfg: SensorConfig):
    """How much noisier this capture is than the bench, as a variance ratio.

    The bench sigmas were measured on a table. On a skater the same sensor sees
    stride impacts, blade chatter and mounting compliance, none of which the
    filter models - so they belong in Q rather than being treated as signal.
    This measures the high-frequency floor of the *moving* part of the capture.

    It is an upper bound: some of that content is real motion of the sensor.
    """
    moving = ~stationary
    if moving.sum() < 100:
        return 1.0, 1.0
    sa = hf_sigma(accel_g[moving] * G0)
    sg = hf_sigma(gyro_dps[moving])
    ratio_a = float(np.median((sa / (cfg.accel_std_g * G0)) ** 2))
    ratio_g = float(np.median((sg / cfg.gyro_std_dps) ** 2))
    return max(ratio_a, 1.0), max(ratio_g, 1.0)


class KFNoise:
    """Process and measurement noise, derived from static_drift_config.json.

    White-noise densities come from the Allan block (ARW/VRW). The bias random
    walks treat each bias as a first-order Markov process and use its
    equivalent random-walk sigma, ``sigma_BI * sqrt(2/tau)``. Initial
    uncertainties come from the bias block's own caveats: the gyro from
    run-to-run repeatability, the accelerometer from the tilt/scale ambiguity a
    single-orientation bench run cannot resolve.
    """

    def __init__(self, cfg: SensorConfig, q_scale: float = 1.0,
                 q_scale_gyro: float | None = None, sigma_zupt: float = 0.05,
                 sigma_tilt_deg: float = 1.0, zaru_k: float = 6.0,
                 sigma_height: float = 0.15, sigma_vspeed: float = 0.15):
        allan = cfg.raw["noise"]["allan"]
        # ARW [deg/sqrt(h)] -> rad/s/sqrt(Hz); VRW [m/s/sqrt(h)] -> m/s^2/sqrt(Hz)
        self.arw = np.array([allan["gyro_arw_deg_sqrt_h"][k] for k in "xyz"]) * DEG / 60.0
        self.vrw = np.array([allan["accel_vrw_ms_sqrt_h"][k] for k in "xyz"]) / 60.0

        bi_g = np.array([allan["gyro_bias_instability_deg_h"][k] for k in "xyz"]) * DEG / 3600.0
        tau_g = np.array([allan["gyro_bias_instability_tau_s"][k] for k in "xyz"])
        bi_a = np.array([allan["accel_bias_instability_mg"][k] for k in "xyz"]) * 1e-3 * G0
        tau_a = np.array([allan["accel_bias_instability_tau_s"][k] for k in "xyz"])
        self.bg_rw = bi_g * np.sqrt(2.0 / tau_g)     # rad/s / sqrt(s)
        self.ba_rw = bi_a * np.sqrt(2.0 / tau_a)     # m/s^2 / sqrt(s)

        self.q_scale = float(q_scale)
        self.q_scale_gyro = float(q_scale if q_scale_gyro is None else q_scale_gyro)

        # initial 1-sigma
        self.p0_pos = 1e-3
        self.p0_vel = 0.01
        self.p0_tilt = 1.0 * DEG        # levelling from a still accel mean
        self.p0_yaw = 180.0 * DEG       # unobservable: say so honestly
        self.p0_bg = _gyro_bias_repeatability(cfg.raw) * DEG
        self.p0_ba = float(cfg.raw["bias"]["accel_norm_error_ms2"])

        # measurement noise
        self.r_zupt = float(sigma_zupt) ** 2
        self.r_zaru = (zaru_k * np.radians(cfg.gyro_std_dps)) ** 2
        self.r_tilt = (sigma_tilt_deg * DEG) ** 2
        # planar: how far the sensor may ride above the ice, and how fast it
        # may move vertically, over a run on a flat rink
        self.r_height = float(sigma_height) ** 2
        self.r_vspeed = float(sigma_vspeed) ** 2

    def describe(self) -> str:
        return (f"ARW {np.round(np.degrees(self.arw) * 60, 4)} deg/sqrt(h), "
                f"VRW {np.round(self.vrw * 60, 5)} m/s/sqrt(h)\n"
                f"Q scale: accel {self.q_scale:g}x, gyro {self.q_scale_gyro:g}x "
                f"(variance, on the white terms)\n"
                f"P0: tilt {self.p0_tilt / DEG:.1f} deg, yaw {self.p0_yaw / DEG:.0f} deg, "
                f"gyro bias {self.p0_bg / DEG:.4f} dps, "
                f"accel bias {self.p0_ba:.4f} m/s^2\n"
                f"R: ZUPT {math.sqrt(self.r_zupt):.3f} m/s, "
                f"ZARU {np.round(np.degrees(np.sqrt(self.r_zaru)), 4)} dps, "
                f"tilt {math.sqrt(self.r_tilt) / DEG:.2f} deg")


# ------------------------------------------------------------------ the filter


@dataclass
class Estimate:
    """Per-sample estimator output."""

    t_s: np.ndarray
    pos: np.ndarray             # (N, 3) m, world, origin at the first sample
    vel: np.ndarray             # (N, 3) m/s, world
    acc: np.ndarray             # (N, 3) m/s^2, world, gravity removed
    euler: np.ndarray           # (N, 3) rad, roll/pitch/yaw
    gyro_world: np.ndarray      # (N, 3) rad/s, bias-corrected, world frame
    accel_bias: np.ndarray      # (N, 3) m/s^2
    gyro_bias: np.ndarray       # (N, 3) rad/s
    sigma: np.ndarray           # (N, 15) 1-sigma of the error state
    stationary: np.ndarray      # (N,) bool, the mask the updates used
    counts: dict = field(default_factory=dict)
    init_euler_deg: tuple = (0.0, 0.0, 0.0)

    @property
    def n(self) -> int:
        return len(self.t_s)

    @property
    def speed_ms(self) -> np.ndarray:
        """Horizontal speed. The vertical channel on flat ice is noise."""
        return np.linalg.norm(self.vel[:, :2], axis=1)

    @property
    def yaw_rate_dps(self) -> np.ndarray:
        """Turn rate about the world vertical - what a skater calls turning."""
        return np.degrees(self.gyro_world[:, 2])

    @property
    def final_pos_sigma_m(self) -> float:
        return float(np.linalg.norm(self.sigma[-1, 0:2]))

    def distance_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.pos[:, :2], axis=0), axis=1).sum())


def initial_attitude(accel_g: np.ndarray, stationary: np.ndarray,
                     bias_g: np.ndarray):
    """Level from the mean specific force over the first still stretch.

    Averaging over the stretch rather than using one sample matters: a single
    sample carries the full noise floor into the initial tilt, and an initial
    tilt error leaks gravity into the horizontal channels for the whole run.
    """
    sel = stationary.copy()
    if sel.sum() < 10:
        # Never still. Fall back to the start and accept a worse initial tilt;
        # the filter's own tilt updates will improve it if the skater ever stops.
        sel = np.zeros(len(stationary), dtype=bool)
        sel[1:11] = True
    first = int(np.flatnonzero(sel)[0])
    window = sel & (np.arange(len(sel)) < first + 500)
    mean = (accel_g[window] - bias_g).mean(axis=0) * G0
    roll, pitch = accel_to_roll_pitch(-mean)
    return roll, pitch, int(window.sum())


def _update(H, y, R_meas, P, p, v, q, ba, bg):
    """One Kalman update: gain, injection into the nominal state, Joseph form.

    Joseph form rather than (I - KH)P: it stays symmetric and positive definite
    under finite precision, and this filter runs tens of thousands of updates.
    """
    S = H @ P @ H.T + R_meas
    K = P @ H.T @ np.linalg.inv(S)
    dx = K @ y

    p = p + dx[0:3]
    v = v + dx[3:6]
    q = quat_mul(quat_from_rotvec(dx[6:9]), q)     # global (world) angle error
    q /= np.linalg.norm(q)
    ba = ba + dx[9:12]
    bg = bg + dx[12:15]

    IKH = np.eye(15) - K @ H
    P = IKH @ P @ IKH.T + K @ R_meas @ K.T
    return p, v, q, ba, bg, P


def run(t_ms: np.ndarray, accel_g: np.ndarray, gyro_dps: np.ndarray,
        stationary: np.ndarray, cfg: SensorConfig, noise: KFNoise,
        gyro_bias_dps: np.ndarray | None = None,
        use_zaru: bool = True, use_tilt: bool = True,
        planar: bool = True) -> Estimate:
    """Strapdown mechanisation plus the ESKF. One pass, no smoothing.

    ``planar`` adds a height and vertical-speed pseudo-measurement. A rink is
    flat, so the sensor cannot climb; without it the vertical channel absorbs
    tilt error and drags the horizontal solution with it through the
    covariance.
    """
    n = len(t_ms)
    t = t_ms / 1000.0
    accel = accel_g * G0
    gyro = np.radians(gyro_dps)

    bias_g = cfg.accel_bias_g
    roll, pitch, n_level = initial_attitude(accel_g, stationary, bias_g)
    q = quat_from_euler(roll, pitch, 0.0)
    p = np.zeros(3)
    v = np.zeros(3)
    ba = bias_g * G0
    bg = np.radians(cfg.gyro_bias_dps if gyro_bias_dps is None else gyro_bias_dps)

    P = np.zeros((15, 15))
    P[0:3, 0:3] = np.eye(3) * noise.p0_pos ** 2
    P[3:6, 3:6] = np.eye(3) * noise.p0_vel ** 2
    P[6:9, 6:9] = np.diag([noise.p0_tilt ** 2, noise.p0_tilt ** 2, noise.p0_yaw ** 2])
    P[9:12, 9:12] = np.eye(3) * noise.p0_ba ** 2
    P[12:15, 12:15] = np.eye(3) * noise.p0_bg ** 2

    out = {k: np.zeros((n, 3)) for k in
           ("pos", "vel", "acc", "euler", "gyro_world", "ba", "bg")}
    sigma = np.zeros((n, 15))
    zupt_mask = np.zeros(n, dtype=bool)
    n_zupt = n_zaru = n_tilt = n_planar = n_gap = 0

    R_mat = quat_to_matrix(q)
    out["euler"][0] = (roll, pitch, 0.0)
    out["ba"][0] = ba
    out["bg"][0] = bg
    out["gyro_world"][0] = R_mat @ (gyro[0] - bg)
    sigma[0] = np.sqrt(np.diag(P))

    eye15 = np.eye(15)
    g_world = np.array([0.0, 0.0, -G0])
    up_hat = np.array([0.0, 0.0, 1.0])

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0.0 or dt > MAX_DT_S:
            # Carry the state across the hole rather than integrating it.
            n_gap += 1
            out["pos"][i], out["vel"][i] = p, v
            out["euler"][i] = matrix_to_euler(R_mat)
            out["gyro_world"][i] = R_mat @ (gyro[i] - bg)
            out["ba"][i], out["bg"][i] = ba, bg
            sigma[i] = np.sqrt(np.diag(P))
            continue

        # ---- nominal propagation ----------------------------------------
        w = gyro[i] - bg
        q = quat_mul(q, quat_from_rotvec(w * dt))
        q /= np.linalg.norm(q)
        R_mat = quat_to_matrix(q)

        f = accel[i] - ba
        a_world = R_mat @ f + g_world
        p = p + v * dt + 0.5 * a_world * dt * dt
        v = v + a_world * dt

        # ---- error-state propagation -------------------------------------
        F = eye15.copy()
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -skew(R_mat @ f) * dt
        F[3:6, 9:12] = -R_mat * dt
        F[6:9, 12:15] = -R_mat * dt

        qa = (noise.vrw ** 2) * noise.q_scale        # (m/s^2)^2 / Hz
        qg = (noise.arw ** 2) * noise.q_scale_gyro   # (rad/s)^2 / Hz
        Q = np.zeros((15, 15))
        Q[0:3, 0:3] = np.diag(qa) * (dt ** 3) / 3.0
        Q[3:6, 3:6] = np.diag(qa) * dt
        Q[6:9, 6:9] = np.diag(qg) * dt
        Q[9:12, 9:12] = np.diag(noise.ba_rw ** 2) * dt
        Q[12:15, 12:15] = np.diag(noise.bg_rw ** 2) * dt
        P = F @ P @ F.T + Q

        # ---- updates, only while standing still --------------------------
        if stationary[i]:
            zupt_mask[i] = True

            if cfg.zupt:
                H = np.zeros((3, 15))
                H[:, 3:6] = np.eye(3)
                p, v, q, ba, bg, P = _update(
                    H, -v, np.eye(3) * noise.r_zupt, P, p, v, q, ba, bg)
                R_mat = quat_to_matrix(q)
                n_zupt += 1

            if use_zaru:
                H = np.zeros((3, 15))
                H[:, 12:15] = np.eye(3)
                p, v, q, ba, bg, P = _update(
                    H, gyro[i] - bg, np.diag(noise.r_zaru), P, p, v, q, ba, bg)
                R_mat = quat_to_matrix(q)
                n_zaru += 1

            if use_tilt:
                fi = accel[i] - ba
                norm = float(np.linalg.norm(fi))
                if norm > 1e-6:
                    f_hat = fi / norm
                    pred = R_mat @ f_hat
                    H = np.zeros((3, 15))
                    H[:, 6:9] = -skew(pred)
                    H[:, 9:12] = -R_mat @ (np.eye(3) - np.outer(f_hat, f_hat)) / norm
                    p, v, q, ba, bg, P = _update(
                        H, up_hat - pred, np.eye(3) * noise.r_tilt,
                        P, p, v, q, ba, bg)
                    R_mat = quat_to_matrix(q)
                    n_tilt += 1

        # ---- the ice is flat: height and vertical speed are bounded -------
        if planar:
            H = np.zeros((2, 15))
            H[0, 2] = 1.0
            H[1, 5] = 1.0
            R_meas = np.diag([noise.r_height, noise.r_vspeed])
            p, v, q, ba, bg, P = _update(
                H, np.array([-p[2], -v[2]]), R_meas, P, p, v, q, ba, bg)
            R_mat = quat_to_matrix(q)
            n_planar += 1

        out["pos"][i], out["vel"][i] = p, v
        out["acc"][i] = a_world
        out["euler"][i] = matrix_to_euler(R_mat)
        out["gyro_world"][i] = R_mat @ (gyro[i] - bg)
        out["ba"][i], out["bg"][i] = ba, bg
        sigma[i] = np.sqrt(np.maximum(np.diag(P), 0.0))

    return Estimate(
        t_s=t - t[0],
        pos=out["pos"],
        vel=out["vel"],
        acc=out["acc"],
        euler=out["euler"],
        gyro_world=out["gyro_world"],
        accel_bias=out["ba"],
        gyro_bias=out["bg"],
        sigma=sigma,
        stationary=zupt_mask,
        counts={"zupt": n_zupt, "zaru": n_zaru, "tilt": n_tilt,
                "planar": n_planar, "gaps": n_gap,
                "levelling_samples": n_level},
        init_euler_deg=(math.degrees(roll), math.degrees(pitch), 0.0),
    )
