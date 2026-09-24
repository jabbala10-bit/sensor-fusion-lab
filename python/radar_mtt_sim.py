#!/usr/bin/env python3
"""Radar multi-target tracking.

Extended targets return several detections, clutter adds false alarms, detections are clustered,
and a GNN tracker keeps tracks: constant-velocity Kalman filters, Mahalanobis gating, Hungarian
assignment, M-of-N confirmation and miss-based deletion. Writes results/radar_mtt.png.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

DT, STEPS = 0.1, 80
SIG_R, SIG_TH = 0.25, np.deg2rad(1.0)   # radar range and azimuth noise (1 sigma)
PD, CLUTTER_RATE = 0.9, 6.0             # detection probability, mean false alarms per scan
GATE = 9.21                             # chi-square, 2 DOF, 99%
SIGMA_A = 2.0                           # CV process noise: acceleration std [m/s^2]
EXTENT_GATE = 8.0                       # no new track within this distance of an existing one
EPS = 6.0                               # clustering distance [m]: must exceed the spacing of
#                                         detections on the longest object, or it splits in two

TARGETS = [  # x, y, vx, vy, turn rate [rad/s], length, width  (radar at origin, looking along +x)
    (20.0, -6.0, 8.0, 1.0, 0.00, 4.5, 1.8),    # car overtaking to the left
    (75.0, 4.0, -6.0, 0.0, 0.00, 4.5, 1.8),    # oncoming car
    (45.0, 18.0, 0.5, -1.4, 0.00, 0.6, 0.6),   # pedestrian crossing
    (30.0, -22.0, 9.0, 3.0, 0.08, 10.0, 2.5),  # truck merging while turning
]


def truth_trajectories():
    traj = []
    for x, y, vx, vy, w, *_ in TARGETS:
        s, out = np.array([x, y, vx, vy]), []
        for _ in range(STEPS):
            out.append(s.copy())
            c, sn = np.cos(w * DT), np.sin(w * DT)
            s[2:] = [c * s[2] - sn * s[3], sn * s[2] + c * s[3]]  # turn the velocity vector
            s[:2] += s[2:] * DT
        traj.append(np.array(out))
    return traj


def polar_noise_to_cart(xy, rng):
    r, th = np.hypot(*xy), np.arctan2(xy[1], xy[0])
    r, th = r + rng.normal(0, SIG_R), th + rng.normal(0, SIG_TH)
    return np.array([r * np.cos(th), r * np.sin(th)])


def scan(traj, k, rng):
    """Detections of one scan: 1-4 per extended target (with probability PD) plus Poisson clutter."""
    dets, src = [], []
    for i, (tr, spec) in enumerate(zip(traj, TARGETS)):
        if rng.random() > PD:
            continue
        length, width = spec[5], spec[6]
        head = tr[k, 2:] / np.linalg.norm(tr[k, 2:])
        normal = np.array([-head[1], head[0]])
        for _ in range(1 + min(3, rng.poisson(length / 3))):  # scatter points over the body
            p = tr[k, :2] + head * rng.uniform(-0.5, 0.5) * length + normal * rng.uniform(-0.5, 0.5) * width
            dets.append(polar_noise_to_cart(p, rng))
            src.append(i)
    for _ in range(rng.poisson(CLUTTER_RATE)):
        r, th = rng.uniform(5, 100), rng.uniform(-np.pi / 3, np.pi / 3)
        dets.append(np.array([r * np.cos(th), r * np.sin(th)]))
        src.append(-1)
    return np.array(dets).reshape(-1, 2), np.array(src)


def dist_to_body(p, state, spec):
    """Distance from a point to the target body (a segment of its length along the heading)."""
    head = state[2:] / np.linalg.norm(state[2:])
    t = np.clip((p - state[:2]) @ head, -spec[5] / 2, spec[5] / 2)
    return np.linalg.norm(p - (state[:2] + t * head))


def cluster(dets, eps=EPS):
    """Single-linkage clustering (DBSCAN with minPts = 1): union of detections closer than EPS."""
    parent = list(range(len(dets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if np.linalg.norm(dets[i] - dets[j]) < eps:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(dets)):
        groups.setdefault(find(i), []).append(i)
    return [dets[g] for g in groups.values()]


def meas_cov(z, n):
    """Polar noise mapped to Cartesian (Jacobian), averaged over n detections, plus extent spread."""
    r, th = np.hypot(*z), np.arctan2(z[1], z[0])
    J = np.array([[np.cos(th), -r * np.sin(th)], [np.sin(th), r * np.cos(th)]])
    return J @ np.diag([SIG_R**2, SIG_TH**2]) @ J.T / n + np.eye(2) * 1.0


F = np.eye(4)
F[0, 2] = F[1, 3] = DT
G = np.array([[DT**2 / 2, 0], [0, DT**2 / 2], [DT, 0], [0, DT]])
Q = G @ G.T * SIGMA_A**2
H = np.eye(2, 4)


class Track:
    next_id = 1

    def __init__(self, z, R):
        self.x = np.r_[z, 0.0, 0.0]
        self.P = np.diag([R[0, 0], R[1, 1], 25.0, 25.0])  # velocity unknown: 5 m/s std
        self.id, self.hits, self.age, self.misses, self.confirmed = Track.next_id, 1, 1, 0, False
        Track.next_id += 1
        self.history = []

    def predict(self):
        self.x, self.P = F @ self.x, F @ self.P @ F.T + Q
        self.age += 1

    def innovation(self, z, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S

    def update(self, z, R):
        y, S = self.innovation(z, R)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1
        self.misses = 0


def run(traj, eps, absorb=False, seed=4):
    """One full pass of the tracker.

    absorb=True: a leftover cluster within EXTENT_GATE metres of an existing track spawns no new
    track. The statistical gate is too tight for this: a second cluster on the same 10 m truck sits
    well outside a 99% gate.
    Long vehicles split into several clusters, and without this each extra cluster starts a
    competing track that starves the real one.
    """
    rng = np.random.default_rng(seed)
    Track.next_id = 1
    tracks, dead, all_dets = [], [], []
    n_conf, n_false, n_clusters, errs, first_conf = [], [], [], [], {}
    for k in range(STEPS):
        dets, src = scan(traj, k, rng)
        all_dets.append((dets, src))
        clusters = cluster(dets, eps) if len(dets) else []
        zs = [c.mean(0) for c in clusters]
        Rs = [meas_cov(z, len(c)) for z, c in zip(zs, clusters)]
        for t in tracks:
            t.predict()
        # Global nearest neighbour: Mahalanobis cost, gate, Hungarian assignment.
        cost = np.full((len(tracks), len(zs)), 1e6)
        for i, t in enumerate(tracks):
            for j, (z, R) in enumerate(zip(zs, Rs)):
                y, S = t.innovation(z, R)
                d2 = y @ np.linalg.solve(S, y)
                if d2 < GATE:
                    cost[i, j] = d2
        rows, cols = linear_sum_assignment(cost) if cost.size else ([], [])
        used_t, used_z = set(), set()
        for i, j in zip(rows, cols):
            if cost[i, j] < 1e6:
                tracks[i].update(zs[j], Rs[j])
                used_t.add(i)
                used_z.add(j)
        for i, t in enumerate(tracks):
            if i not in used_t:
                t.misses += 1
        existing = list(tracks)
        for j, (z, R) in enumerate(zip(zs, Rs)):
            if j in used_z:
                continue
            if absorb and any(np.linalg.norm(t.x[:2] - z) < EXTENT_GATE for t in existing):
                continue                           # extra return from an object already tracked
            tracks.append(Track(z, R))
        # Track management: confirm 3 hits within the first 5 scans; delete after misses.
        keep = []
        for t in tracks:
            if not t.confirmed and t.hits >= 3 and t.age <= 5:
                t.confirmed = True
            drop = (not t.confirmed and (t.misses >= 2 or t.age > 5)) or (t.confirmed and t.misses >= 4)
            (dead if drop else keep).append(t)
            if not drop:
                t.history.append((k, *t.x[:2]))
        tracks = keep
        conf = [t for t in tracks if t.confirmed]
        false = 0
        for t in conf:
            d = np.array([dist_to_body(t.x[:2], tr[k], spec) for tr, spec in zip(traj, TARGETS)])
            if d.min() < 3.0:                      # inside or beside a real object
                errs.append(d.min())
                first_conf.setdefault(int(np.argmin(d)), k)
            else:
                false += 1
        n_conf.append(len(conf))
        n_false.append(false)
        n_clusters.append(len(clusters))

    ever = [t for t in tracks + dead if t.confirmed]
    return dict(dets=all_dets, tracks=ever, n_conf=n_conf, n_false=n_false, n_clusters=n_clusters,
                errs=errs, first=first_conf, started=Track.next_id - 1)


def main():
    os.makedirs("results", exist_ok=True)
    traj = truth_trajectories()
    runs = {("3 m, one cluster per track"): run(traj, 3.0),
            ("6 m, one cluster per track"): run(traj, 6.0),
            ("6 m, extra clusters absorbed"): run(traj, 6.0, absorb=True)}
    r = runs["6 m, extra clusters absorbed"]
    n_dets = sum(len(d) for d, _ in r["dets"])
    n_clutter = sum(int((s == -1).sum()) for _, s in r["dets"])
    print(f"{STEPS} scans | {n_dets} detections ({n_clutter} clutter) | 4 targets, the longest 10 m\n")
    print(f"{'setting':>30} {'clusters/scan':>14} {'started':>8} {'confirmed':>10} "
          f"{'conf/scan':>10} {'false track-scans':>18} {'RMSE m':>8}")
    for name, d in runs.items():
        print(f"{name:>30} {np.mean(d['n_clusters']):14.1f} {d['started']:8d} {len(d['tracks']):10d} "
              f"{np.mean(d['n_conf'][10:]):10.1f} {sum(d['n_false']):18d} "
              f"{np.sqrt(np.mean(np.square(d['errs']))):8.2f}")
    print(f"\nbest setting: confirmed per scan after scan 10 min {min(r['n_conf'][10:])}, max {max(r['n_conf'][10:])} "
          f"(truth 4) | first confirmation at scan {dict(sorted(r['first'].items()))}")
    all_dets, ever, n_conf, n_false = r["dets"], r["tracks"], r["n_conf"], r["n_false"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for dets, src in all_dets:
        a.scatter(dets[src == -1, 0], dets[src == -1, 1], s=3, c="0.75")
        a.scatter(dets[src >= 0, 0], dets[src >= 0, 1], s=3, c="0.45")
    for tr in traj:
        a.plot(tr[:, 0], tr[:, 1], "k-", lw=3, alpha=0.3)
    cmap = plt.get_cmap("tab10")
    for n, t in enumerate(sorted(ever, key=lambda t: t.id)):
        h = np.array(t.history)
        a.plot(h[:, 1], h[:, 2], "-", color=cmap(n % 10), lw=1.6, label=f"track {t.id}")
        a.annotate(f"#{t.id}", h[-1, 1:], fontsize=8, color=cmap(n % 10))
    a.scatter([0], [0], marker="^", c="r", s=60)
    a.set(xlabel="x [m]", ylabel="y [m]", title="Clustering 6 m with absorption: detections (grey), truth (thick), confirmed tracks")
    a.set_aspect("equal")
    a.legend(fontsize=7, ncol=2, loc="lower right")
    t_axis = np.arange(STEPS) * DT
    b.step(t_axis, n_conf, where="post", label="confirmed tracks")
    b.step(t_axis, n_false, where="post", label="false confirmed tracks")
    b.axhline(4, color="k", ls=":", label="true targets")
    b.set(xlabel="time [s]", ylabel="count", title="Track management over time")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/radar_mtt.png", dpi=105)
    print("-> results/radar_mtt.png")


if __name__ == "__main__":
    main()
