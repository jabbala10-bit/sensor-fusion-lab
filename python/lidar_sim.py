#!/usr/bin/env python3
"""Lidar obstacle detection simulation.

Ray-cast lidar -> voxel grid -> ROI crop -> RANSAC ground plane (+ SVD refinement)
-> own KD-tree -> Euclidean clustering (fixed vs range-adaptive) -> AABB, PCA, min-area and L-shape boxes.
Writes results/lidar_bev.png (bird's-eye view + range image) and prints a cluster table.
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

H = 1.73  # sensor height [m]; ground is the plane z = -H in the sensor frame
# name: (centre x, centre y, length, width, height, yaw [deg]) - sensor frame, x fwd, y left
OBJECTS = {
    "car_ahead": (14.25, 0.0, 4.5, 1.8, 1.5, 0),
    "car_left": (8.25, 3.5, 4.5, 1.8, 1.5, 0),
    "truck_right": (24.0, -3.45, 8.0, 2.5, 3.2, 0),
    "car_behind": (-11.75, 0.0, 4.5, 1.8, 1.5, 0),
    "pedestrian": (9.25, 6.75, 0.5, 0.5, 1.8, 0),
    "pole": (15.15, -6.35, 0.3, 0.3, 4.0, 0),
    "car_merging": (27.0, 5.0, 4.5, 1.8, 1.5, 20),  # rotated: shows why AABB is not enough
}
EGO_ROOF = (-0.25, 0.0, 2.5, 1.7, H - 0.4, 0)  # roof top 0.4 m below the sensor


def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def simulate_scan(rng, layers=32, elev=(-24.8, 2.0), az_res=0.4, rmin=0.3, rmax=60.0, sigma=0.02):
    """Cast one ray per (layer, azimuth); nearest hit of ground or box wins. Returns points and range image."""
    el = np.deg2rad(np.linspace(*elev, layers))[:, None]
    az = np.deg2rad(-180.0 + az_res * np.arange(int(round(360 / az_res))))[None, :]
    d = np.stack(np.broadcast_arrays(np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1)
    t = np.where(d[..., 2] < 0, -H / np.minimum(d[..., 2], -1e-12), np.inf)  # ground hit distance
    with np.errstate(divide="ignore", invalid="ignore"):
        for cx, cy, length, width, height, yaw in list(OBJECTS.values()) + [EGO_ROOF]:
            # Transform the rays into the box frame, then use the slab test on half-extents.
            r = rot2(np.deg2rad(yaw))
            o = np.array([-cx, -cy]) @ r  # ray origin (sensor at 0,0) in box frame
            dxy = d[..., :2] @ r
            lo = np.array([-length / 2 - o[0], -width / 2 - o[1], -H])
            hi = np.array([length / 2 - o[0], width / 2 - o[1], -H + height])
            dl = np.concatenate([dxy, d[..., 2:]], -1)
            t1, t2 = lo / dl, hi / dl
            tmin = np.max(np.minimum(t1, t2), -1)
            tmax = np.min(np.maximum(t1, t2), -1)
            t = np.where((tmin <= tmax) & (tmin > 0) & (tmin < t), tmin, t)
    rng_meas = t + rng.normal(0.0, sigma, t.shape)
    valid = np.isfinite(t) & (t >= rmin) & (t <= rmax)
    range_image = np.where(valid, rng_meas, np.nan)
    return d[valid] * rng_meas[valid][:, None], range_image


def voxel_grid(p, leaf):
    """Replace all points inside each leaf^3 voxel by their centroid."""
    _, inv, counts = np.unique(np.floor(p / leaf).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.size, 3))
    np.add.at(sums, inv.ravel(), p)
    return sums / counts[:, None]


def crop(p, lo, hi, ego_lo, ego_hi):
    keep = np.all((p >= lo) & (p <= hi), 1) & ~np.all((p >= ego_lo) & (p <= ego_hi), 1)
    return p[keep]


def ransac_plane(p, rng, max_iter=200, tol=0.2, max_tilt_deg=15.0, prob=0.99):
    """RANSAC with a ground prior (normal near vertical) and an adaptive iteration count."""
    best, model, needed, it = np.zeros(0, int), None, max_iter, 0
    min_nz = np.cos(np.deg2rad(max_tilt_deg))
    while it < needed:
        it += 1
        p1, p2, p3 = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-6 or abs(n[2]) / norm < min_nz:
            continue  # degenerate sample, or a wall rather than the road
        n /= norm
        inl = np.flatnonzero(np.abs(p @ n - n @ p1) <= tol)
        if inl.size > best.size:
            best, model = inl, (n, -n @ p1)
            w = inl.size / len(p)
            needed = it if w >= 1 else min(max_iter, int(np.ceil(np.log(1 - prob) / np.log(1 - w**3))))
    return model, best, it


def refine_plane_svd(p):
    """Total least squares: normal = right singular vector of the smallest singular value."""
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    return n, -n @ c


class KDTree:
    """3-D KD-tree built by median split; radius search with a cheap box test first."""

    def __init__(self, pts):
        self.p = pts.tolist()
        self.nodes = []  # [point index, left, right, axis]
        self.root = self._build(list(range(len(pts))), 0)

    def _build(self, ids, depth):
        if not ids:
            return -1
        axis = depth % 3
        ids.sort(key=lambda i: self.p[i][axis])
        mid = len(ids) // 2
        node = len(self.nodes)
        self.nodes.append([ids[mid], -1, -1, axis])
        self.nodes[node][1] = self._build(ids[:mid], depth + 1)
        self.nodes[node][2] = self._build(ids[mid + 1:], depth + 1)
        return node

    def radius_search(self, q, r):
        out, stack, r2 = [], [self.root], r * r
        while stack:
            n = stack.pop()
            if n < 0:
                continue
            idx, left, right, axis = self.nodes[n]
            pt = self.p[idx]
            dx, dy, dz = pt[0] - q[0], pt[1] - q[1], pt[2] - q[2]
            if abs(dx) <= r and abs(dy) <= r and abs(dz) <= r and dx * dx + dy * dy + dz * dz <= r2:
                out.append(idx)
            if q[axis] - r < pt[axis]:
                stack.append(left)
            if q[axis] + r > pt[axis]:
                stack.append(right)
        return out


def euclidean_cluster(p, tree, tol_min, tol_per_m=0.0, min_size=5, max_size=5000):
    """Region growing: tolerance = max(tol_min, tol_per_m * horizontal range)."""
    processed = np.zeros(len(p), bool)
    rxy = np.hypot(p[:, 0], p[:, 1])
    pl, clusters = p.tolist(), []
    for i in range(len(p)):
        if processed[i]:
            continue
        stack, cluster = [i], []
        processed[i] = True
        while stack:
            j = stack.pop()
            cluster.append(j)
            for k in tree.radius_search(pl[j], max(tol_min, tol_per_m * rxy[j])):
                if not processed[k]:
                    processed[k] = True
                    stack.append(k)
        if min_size <= len(cluster) <= max_size:
            clusters.append(np.array(cluster))
    return clusters


def pca_yaw(xy):
    evals, evecs = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
    major = evecs[:, np.argmax(evals)]
    return np.arctan2(major[1], major[0])


def fit_box(xy, criterion="closeness", step_deg=0.5, d0=0.01):
    """Search yaw in [0, 90) and keep the best rectangle.

    criterion="area":      smallest enclosing rectangle (fails on L-shapes whose hull is a triangle)
    criterion="closeness": L-shape fitting - reward points that lie close to a rectangle edge
    """
    best = None
    for yaw in np.deg2rad(np.arange(0.0, 90.0, step_deg)):
        local = xy @ rot2(yaw)  # points expressed in the rotated frame
        lo, hi = local.min(0), local.max(0)
        if criterion == "area":
            score = -np.prod(np.maximum(hi - lo, 0.05))
        else:
            dist = np.minimum(local - lo, hi - local)  # distance to the nearer edge, per axis
            score = np.sum(1.0 / np.maximum(dist.min(1), d0))
        if best is None or score > best[0]:
            best = (score, yaw, lo, hi)
    _, yaw, lo, hi = best
    size = hi - lo
    centre = rot2(yaw) @ ((lo + hi) / 2)
    if size[1] > size[0]:  # length = longer side
        size, yaw = size[::-1], yaw + np.pi / 2
    return centre, size, (yaw + np.pi / 2) % np.pi - np.pi / 2  # yaw modulo 180 deg


def yaw_err_deg(est, true_deg):
    """Rectangle yaw is ambiguous by 90 deg, so compare modulo 90."""
    return (np.degrees(est) - true_deg + 45.0) % 90.0 - 45.0


def rect_corners(centre, size, yaw):
    half = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]) * size / 2
    return half @ rot2(yaw).T + centre


def main():
    rng = np.random.default_rng(7)
    os.makedirs("results", exist_ok=True)
    t0 = time.perf_counter()
    raw, range_image = simulate_scan(rng)
    vox = voxel_grid(raw, 0.15)
    roi = crop(vox, np.array([-20, -8, -2.5]), np.array([40, 8, 1.5]),
               np.array([-1.7, -1.0, -2.0]), np.array([1.2, 1.0, 0.0]))
    t1 = time.perf_counter()
    (n, d), inl, iters = ransac_plane(roi, rng)
    n_ls, d_ls = refine_plane_svd(roi[inl])
    ground = np.abs(roi @ n_ls + d_ls) <= 0.2
    obst = roi[~ground]
    t2 = time.perf_counter()
    tree = KDTree(obst)
    fixed = euclidean_cluster(obst, tree, tol_min=0.6)
    adaptive = euclidean_cluster(obst, tree, tol_min=0.5, tol_per_m=0.09)
    t3 = time.perf_counter()

    print(f"points  raw {len(raw)} | voxel(0.15 m) {len(vox)} | ROI + ego crop {len(roi)}")
    print(f"RANSAC  {iters} iterations -> {inl.size} inliers; SVD-refined ground height "
          f"{-d_ls / n_ls[2]:.3f} m (true {-H:.3f}), tilt {np.degrees(np.arccos(n_ls[2])):.2f} deg")
    print(f"cluster fixed 0.6 m -> {len(fixed)} clusters | adaptive max(0.5, 0.09 r) -> {len(adaptive)} clusters\n")
    print("yaw error vs truth (deg, modulo 90): PCA | min-area | L-shape closeness")
    print(f"{'object':12s} {'pts':>4s} {'AABB m2':>8s} {'L-shape m2':>10s} {'PCA':>7s} {'minArea':>8s} {'L-shape':>8s}")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1.9, 1]})
    ax.scatter(roi[ground, 0], roi[ground, 1], s=1, c="0.8", label="ground (RANSAC)")
    for name, (cx, cy, length, width, _, yaw) in OBJECTS.items():
        c = rect_corners(np.array([cx, cy]), np.array([length, width]), np.deg2rad(yaw))
        ax.plot(c[:, 0], c[:, 1], "k-", lw=0.8)
    cmap = plt.get_cmap("tab10")
    for k, idx in enumerate(adaptive):
        pts = obst[idx]
        xy = pts[:, :2]
        centre, size, yaw = fit_box(xy, "closeness")
        yaw_area = fit_box(xy, "area")[2]
        aabb = np.prod(xy.max(0) - xy.min(0))
        match = min(OBJECTS, key=lambda o: np.hypot(OBJECTS[o][0] - centre[0], OBJECTS[o][1] - centre[1]))
        true_yaw = OBJECTS[match][5]
        print(f"{match:12s} {len(idx):4d} {aabb:8.1f} {np.prod(size):10.1f} {yaw_err_deg(pca_yaw(xy), true_yaw):7.1f} "
              f"{yaw_err_deg(yaw_area, true_yaw):8.1f} {yaw_err_deg(yaw, true_yaw):8.1f}")
        col = cmap(k % 10)
        ax.scatter(xy[:, 0], xy[:, 1], s=4, color=col)
        lo, hi = xy.min(0), xy.max(0)
        ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]], ":", color=col, lw=1)
        c = rect_corners(centre, size, yaw)
        ax.plot(c[:, 0], c[:, 1], "-", color=col, lw=1.8)
        ax.annotate(match, centre + np.array([0, 1.2]), ha="center", fontsize=8, color=col)
    ax.plot([], [], "k-", lw=0.8, label="ground truth")
    ax.plot([], [], "k:", label="AABB")
    ax.plot([], [], "k-", lw=1.8, label="L-shape box")
    ax.scatter([0], [0], marker="^", c="r", s=60, label="sensor")
    ax.set(xlim=(-20, 40), ylim=(-8, 8), xlabel="x forward [m]", ylabel="y left [m]",
           title=f"Bird's-eye view: {len(adaptive)} clusters (fixed tolerance gave {len(fixed)})")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    im = ax2.imshow(range_image, aspect="auto", origin="lower", cmap="viridis", extent=[-180, 180, 0, 32])
    ax2.set(xlabel="azimuth [deg]", ylabel="beam (layer)", title="Range image representation (32 x 900)")
    fig.colorbar(im, ax=ax2, label="range [m]")
    fig.tight_layout()
    fig.savefig("results/lidar_bev.png", dpi=110)
    print(f"\ntiming  simulate+filter {1e3 * (t1 - t0):.0f} ms | segment {1e3 * (t2 - t1):.0f} ms | "
          f"cluster x2 {1e3 * (t3 - t2):.0f} ms  -> results/lidar_bev.png")


if __name__ == "__main__":
    main()
