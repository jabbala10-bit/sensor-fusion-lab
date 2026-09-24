#!/usr/bin/env python3
"""Lidar + radar fusion: KF (lidar only, CV), EKF (fusion, CV) and UKF (fusion, CTRV).

Generates a CTRV ground-truth trajectory, simulates alternating lidar and radar measurements,
runs the three filters, reports RMSE, checks NIS consistency and sweeps the UKF process noise.
Writes results/kalman_fusion.png.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DT_TRUTH, DURATION, DT_MEAS = 0.002, 25.0, 0.05
S_LIDAR, S_RHO, S_PHI, S_RHODOT = 0.15, 0.3, 0.03, 0.3
R_LIDAR = np.diag([S_LIDAR**2, S_LIDAR**2])
R_RADAR = np.diag([S_RHO**2, S_PHI**2, S_RHODOT**2])
CHI2_95 = {2: 5.991, 3: 7.815}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def trajectory():
    """CTRV with time-varying yaw rate and acceleration; returns t, px, py, vx, vy."""
    px, py, v, psi, out = -5.0, -15.0, 5.0, 0.0, []
    for t in np.arange(0, DURATION + 1e-9, DT_TRUTH):
        out.append((t, px, py, v * np.cos(psi), v * np.sin(psi)))
        acc, yaw_rate = 0.6 * np.sin(0.5 * t), 0.25 + 0.15 * np.sin(0.4 * t)
        px += v / yaw_rate * (np.sin(psi + yaw_rate * DT_TRUTH) - np.sin(psi))
        py += v / yaw_rate * (-np.cos(psi + yaw_rate * DT_TRUTH) + np.cos(psi))
        psi = wrap(psi + yaw_rate * DT_TRUTH)
        v += acc * DT_TRUTH
    return np.array(out)


def measurements(truth, rng):
    """Alternating lidar (px, py) and radar (rho, phi, rho_dot) at 20 Hz total."""
    out = []
    for k in range(int(DURATION / DT_MEAS) + 1):
        t, px, py, vx, vy = truth[int(round(k * DT_MEAS / DT_TRUTH))]
        if k % 2 == 0:
            z = np.array([px, py]) + rng.normal(0, S_LIDAR, 2)
            out.append((t, "lidar", z, (px, py, vx, vy)))
        else:
            rho = np.hypot(px, py)
            z = np.array([rho, np.arctan2(py, px), (px * vx + py * vy) / rho])
            z += rng.normal(0, [S_RHO, S_PHI, S_RHODOT])
            out.append((t, "radar", z, (px, py, vx, vy)))
    return out


class KalmanCV:
    """Linear KF with a constant-velocity model; EKF update for the nonlinear radar model."""

    def __init__(self, z, sigma_a=3.0):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([S_LIDAR**2, S_LIDAR**2, 100.0, 100.0])
        self.sigma_a = sigma_a

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.sigma_a**2

    def update_lidar(self, z):
        H = np.eye(2, 4)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R_LIDAR
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return y @ np.linalg.solve(S, y)

    def update_radar(self, z):
        px, py, vx, vy = self.x
        c1 = max(1e-6, px * px + py * py)
        c2, c3 = np.sqrt(c1), c1 * np.sqrt(c1)
        h = np.array([c2, np.arctan2(py, px), (px * vx + py * vy) / c2])
        Hj = np.array([[px / c2, py / c2, 0, 0],
                       [-py / c1, px / c1, 0, 0],
                       [py * (vx * py - vy * px) / c3, px * (vy * px - vx * py) / c3, px / c2, py / c2]])
        y = z - h
        y[1] = wrap(y[1])
        S = Hj @ self.P @ Hj.T + R_RADAR
        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ Hj) @ self.P
        return y @ np.linalg.solve(S, y)


class UKF:
    """Unscented Kalman filter with the CTRV model: state px, py, v, psi, psi_dot."""

    N, N_AUG = 5, 7

    def __init__(self, z, std_a=1.0, std_yawdd=0.5):
        self.x = np.array([z[0], z[1], 0.0, 0.0, 0.0])
        self.P = np.diag([S_LIDAR**2, S_LIDAR**2, 25.0, 1.0, 1.0])
        self.std_a, self.std_yawdd = std_a, std_yawdd
        self.lam = 3 - self.N_AUG
        n_sig = 2 * self.N_AUG + 1
        self.w = np.full(n_sig, 0.5 / (self.lam + self.N_AUG))
        self.w[0] = self.lam / (self.lam + self.N_AUG)
        self.Xsig = np.zeros((self.N, n_sig))

    def predict(self, dt):
        xa = np.r_[self.x, 0.0, 0.0]                       # augment with the two process noises
        Pa = np.zeros((self.N_AUG, self.N_AUG))
        Pa[: self.N, : self.N] = self.P
        Pa[5, 5], Pa[6, 6] = self.std_a**2, self.std_yawdd**2
        L = np.linalg.cholesky(Pa)                         # matrix square root
        s = np.sqrt(self.lam + self.N_AUG)
        Xa = np.vstack([xa, xa + s * L.T, xa - s * L.T]).T   # 2 n_aug + 1 sigma points
        px, py, v, psi, psid, nu_a, nu_psi = Xa
        turning = np.abs(psid) > 1e-4
        ppx = np.where(turning, px + v / np.where(turning, psid, 1) * (np.sin(psi + psid * dt) - np.sin(psi)),
                       px + v * np.cos(psi) * dt)
        ppy = np.where(turning, py + v / np.where(turning, psid, 1) * (-np.cos(psi + psid * dt) + np.cos(psi)),
                       py + v * np.sin(psi) * dt)
        self.Xsig = np.vstack([ppx + 0.5 * dt**2 * np.cos(psi) * nu_a,
                               ppy + 0.5 * dt**2 * np.sin(psi) * nu_a,
                               v + dt * nu_a,
                               psi + psid * dt + 0.5 * dt**2 * nu_psi,
                               psid + dt * nu_psi])
        self.x = self.Xsig @ self.w
        self.x[3] = wrap(self.x[3])
        d = self.Xsig - self.x[:, None]
        d[3] = wrap(d[3])
        self.P = (self.w * d) @ d.T

    def update(self, z, R, h, angle_idx=None):
        Zsig = h(self.Xsig)
        z_pred = Zsig @ self.w
        dz = Zsig - z_pred[:, None]
        if angle_idx is not None:
            dz[angle_idx] = wrap(dz[angle_idx])
        S = (self.w * dz) @ dz.T + R
        dx = self.Xsig - self.x[:, None]
        dx[3] = wrap(dx[3])
        T = (self.w * dx) @ dz.T                           # state / measurement cross-correlation
        K = T @ np.linalg.inv(S)
        y = z - z_pred
        if angle_idx is not None:
            y[angle_idx] = wrap(y[angle_idx])
        self.x = self.x + K @ y
        self.x[3] = wrap(self.x[3])
        self.P = self.P - K @ S @ K.T
        return y @ np.linalg.solve(S, y)

    @property
    def cartesian(self):
        return np.array([self.x[0], self.x[1], self.x[2] * np.cos(self.x[3]), self.x[2] * np.sin(self.x[3])])


def h_lidar(X):
    return X[:2]


def h_radar(X):
    px, py, v, psi = X[0], X[1], X[2], X[3]
    rho = np.maximum(1e-6, np.hypot(px, py))
    return np.vstack([rho, np.arctan2(py, px), (px * np.cos(psi) * v + py * np.sin(psi) * v) / rho])


def run(meas, std_a=1.0, std_yawdd=0.5):
    kf = ekf = ukf = None
    rows = []
    t_prev = meas[0][0]
    for t, kind, z, truth in meas:
        if kf is None:
            kf, ekf = KalmanCV(z), KalmanCV(z)
            ukf = UKF(z, std_a, std_yawdd)
            t_prev = t
            continue
        dt = t - t_prev
        t_prev = t
        for f in (kf, ekf, ukf):
            f.predict(dt)
        nis = np.nan
        if kind == "lidar":
            kf.update_lidar(z)
            ekf.update_lidar(z)
            nis = ukf.update(z, R_LIDAR, h_lidar)
        else:
            ekf.update_radar(z)
            nis = ukf.update(z, R_RADAR, h_radar, angle_idx=1)
        rows.append((t, kind, np.array(truth), kf.x.copy(), ekf.x.copy(), ukf.cartesian, nis, z))
    return rows


def rmse(rows, idx):
    err = np.array([r[idx] - r[2] for r in rows])
    return np.sqrt((err**2).mean(0))


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(21)
    truth = trajectory()
    meas = measurements(truth, rng)
    rows = run(meas)
    print(f"{len(meas)} measurements over {DURATION:.0f} s, range from sensor "
          f"{np.hypot(truth[:, 1], truth[:, 2]).min():.1f} to {np.hypot(truth[:, 1], truth[:, 2]).max():.1f} m\n")
    print(f"{'filter':<24}{'RMSE px':>9}{'py':>8}{'vx':>8}{'vy':>8}")
    for name, idx in (("KF  (lidar only, CV)", 3), ("EKF (lidar+radar, CV)", 4), ("UKF (lidar+radar, CTRV)", 5)):
        e = rmse(rows, idx)
        print(f"{name:<24}{e[0]:9.3f}{e[1]:8.3f}{e[2]:8.3f}{e[3]:8.3f}")

    nis_l = np.array([r[6] for r in rows if r[1] == "lidar"])
    nis_r = np.array([r[6] for r in rows if r[1] == "radar"])
    print(f"\nUKF NIS above the 95% line: lidar {100 * (nis_l > CHI2_95[2]).mean():.1f}%, "
          f"radar {100 * (nis_r > CHI2_95[3]).mean():.1f}% (5% expected)")

    print(f"\nProcess-noise tuning sweep\n{'std_a':>6}{'std_yawdd':>11}{'RMSE v':>9}{'lidar NIS>95%':>15}{'radar NIS>95%':>15}")
    for std_a, std_yawdd in ((0.2, 0.1), (1.0, 0.5), (5.0, 2.0)):
        r2 = run(meas, std_a, std_yawdd)
        e = rmse(r2, 5)
        l2 = np.array([r[6] for r in r2 if r[1] == "lidar"])
        rr = np.array([r[6] for r in r2 if r[1] == "radar"])
        print(f"{std_a:6.1f}{std_yawdd:11.1f}{np.hypot(e[2], e[3]):9.3f}"
              f"{100 * (l2 > CHI2_95[2]).mean():14.1f}%{100 * (rr > CHI2_95[3]).mean():14.1f}%")

    # ------------------------------------------------------------------ figure
    t = np.array([r[0] for r in rows])
    tru = np.array([r[2] for r in rows])
    est = {k: np.array([r[i] for r in rows]) for k, i in (("KF lidar only", 3), ("EKF fusion", 4), ("UKF CTRV", 5))}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    lid = np.array([r[7] for r in rows if r[1] == "lidar"])
    rad = np.array([[r[7][0] * np.cos(r[7][1]), r[7][0] * np.sin(r[7][1])] for r in rows if r[1] == "radar"])
    a.scatter(*lid.T, s=6, c="0.7", label="lidar measurements")
    a.scatter(*rad.T, s=6, c="tab:orange", alpha=0.5, label="radar, converted to x-y")
    a.plot(tru[:, 0], tru[:, 1], "k-", lw=2.5, alpha=0.6, label="truth")
    a.plot(est["EKF fusion"][:, 0], est["EKF fusion"][:, 1], "b-", lw=1, label="EKF")
    a.plot(est["UKF CTRV"][:, 0], est["UKF CTRV"][:, 1], "g-", lw=1, label="UKF")
    a.scatter([0], [0], marker="^", c="r", s=70, label="sensors")
    a.set(xlabel="x [m]", ylabel="y [m]", title="Trajectory: radar bearing noise spreads with range")
    a.set_aspect("equal")
    a.legend(fontsize=8)
    a = ax[0, 1]
    a.plot(t, np.hypot(tru[:, 2], tru[:, 3]), "k-", lw=2, label="truth")
    for name, c in (("KF lidar only", "0.5"), ("EKF fusion", "b"), ("UKF CTRV", "g")):
        a.plot(t, np.hypot(est[name][:, 2], est[name][:, 3]), c, lw=1, label=name)
    a.set(xlabel="time [s]", ylabel="speed [m/s]", title="Speed estimate (the hard part for lidar alone)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a = ax[1, 0]
    for name, c in (("KF lidar only", "0.5"), ("EKF fusion", "b"), ("UKF CTRV", "g")):
        a.plot(t, np.linalg.norm(est[name][:, :2] - tru[:, :2], axis=1), c, lw=1, label=name)
    a.set(xlabel="time [s]", ylabel="position error [m]", title="Position error over time")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a = ax[1, 1]
    tl = np.array([r[0] for r in rows if r[1] == "lidar"])
    tr_ = np.array([r[0] for r in rows if r[1] == "radar"])
    a.plot(tl, nis_l, ".-", lw=0.6, ms=3, label="UKF lidar NIS (2 DOF)")
    a.plot(tr_, nis_r, ".-", lw=0.6, ms=3, label="UKF radar NIS (3 DOF)")
    a.axhline(CHI2_95[2], color="tab:blue", ls="--", label="5.991 (95%, 2 DOF)")
    a.axhline(CHI2_95[3], color="tab:orange", ls="--", label="7.815 (95%, 3 DOF)")
    a.set(xlabel="time [s]", ylabel="NIS", ylim=(0, 16), title="NIS consistency check")
    a.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("results/kalman_fusion.png", dpi=105)
    print("\n-> results/kalman_fusion.png")


if __name__ == "__main__":
    main()
