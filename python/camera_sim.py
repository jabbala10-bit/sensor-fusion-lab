#!/usr/bin/env python3
"""Camera simulation.

1. Pinhole projection of lidar points into a synthetic KITTI-sized image.
2. Filtering and gradients: Gaussian blur, Sobel gradient magnitude.
3. Harris corners from scratch (numpy/scipy), checked against cv2.cornerHarris.
4. SIFT keypoints + ratio-test matching -> camera TTC; lidar TTC with min vs median.
5. YOLO-style post-processing: confidence threshold, IoU, non-maximum suppression.
Writes results/camera_sim.png.
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

F, CX, CY, W, H = 1200.0, 621.0, 187.0, 1242, 375  # pinhole intrinsics, KITTI-sized image
Z_OFF = -0.08  # camera sits 8 cm below the lidar, same x and y


def project(p):
    """Lidar frame (x fwd, y left, z up) -> camera frame (X right, Y down, Z fwd) -> pixels."""
    X, Y, Z = -p[:, 1], -(p[:, 2] - Z_OFF), p[:, 0]
    return np.stack([F * X / Z + CX, F * Y / Z + CY], 1)


def car_texture(rng):
    """Rear of a car: 720 x 560 px = 1.8 m x 1.4 m (400 px per metre)."""
    t = np.full((560, 720), 85, np.uint8)
    cv2.rectangle(t, (70, 40), (650, 230), 35, -1)  # rear window
    cv2.rectangle(t, (25, 260), (190, 340), 190, -1)  # tail lights
    cv2.rectangle(t, (530, 260), (695, 340), 190, -1)
    cv2.rectangle(t, (250, 345), (470, 415), 235, -1)  # licence plate
    cv2.putText(t, "KA01 SF26", (262, 397), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 4)
    cv2.circle(t, (360, 290), 28, 220, 5)
    cv2.rectangle(t, (0, 440), (720, 560), 55, -1)  # bumper
    for _ in range(120):  # stickers and dirt give texture for keypoints
        c = rng.integers(0, [720, 560])
        r = int(rng.integers(5, 16))
        cv2.rectangle(t, tuple(int(v) for v in c - r), (int(c[0] + r), int(c[1] + r // 2)), int(rng.integers(30, 235)), -1)
    return t


def background():
    bg = np.empty((H, W), np.uint8)
    bg[: int(CY)] = (210 - np.arange(int(CY)) // 3)[:, None]
    bg[int(CY):] = 105
    for y in (-1.8, 1.8):  # lane dashes converge to the principal point
        for x in np.arange(4.0, 80.0, 6.0):
            a, b = project(np.array([[x, y, -1.73], [x + 3, y, -1.73]]))
            cv2.line(bg, tuple(int(v) for v in a), tuple(int(v) for v in b), 235, max(1, int(12 / x)))
    return bg


def render(tex, bg, d, rng):
    """Warp the texture to the image of a plane at distance d; return frame and car box (x0, y0, x1, y1)."""
    (u0, v0), (u1, v1) = project(np.array([[d, 0.9, -0.03], [d, -0.9, -1.43]]))
    s = (u1 - u0) / tex.shape[1]
    blurred = cv2.GaussianBlur(tex, (0, 0), 0.45 / s)  # anti-aliasing before downscaling
    frame = bg.copy()
    cv2.warpAffine(blurred, np.array([[s, 0, u0], [0, s, v0]]), (W, H), frame,
                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    noisy = frame.astype(np.float32) + rng.normal(0, 2.0, frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8), np.array([u0, v0, u1, v1])


def lidar_rear_face(d, rng, n_outliers=2):
    z, y = np.meshgrid(np.arange(-0.10, -1.41, -0.18), np.arange(-0.85, 0.851, 0.1), indexing="ij")
    pts = np.stack([d + rng.normal(0, 0.02, z.size), y.ravel(), z.ravel()], 1)
    ghosts = np.stack([d - rng.uniform(0.3, 1.0, n_outliers), rng.uniform(-0.8, 0.8, n_outliers),
                       rng.uniform(-1.3, -0.2, n_outliers)], 1)  # spray / exhaust returns
    return np.vstack([pts, ghosts])


def shrink(box, f):
    dx, dy = (box[2] - box[0]) * f / 2, (box[3] - box[1]) * f / 2
    return box + np.array([dx, dy, -dx, -dy])


def inside(pts, box):
    return (pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])


# ------------------------------------------------------------------ Harris from scratch
def harris_response(img, block=3, k=0.04):
    """Same formulation as cv2.cornerHarris: 3x3 Sobel gradients, box window, R = det - k trace^2."""
    I = img.astype(np.float64)
    Ix = ndimage.sobel(I, axis=1, mode="mirror")
    Iy = ndimage.sobel(I, axis=0, mode="mirror")
    Sxx = ndimage.uniform_filter(Ix * Ix, block, mode="mirror")  # structure tensor entries
    Syy = ndimage.uniform_filter(Iy * Iy, block, mode="mirror")
    Sxy = ndimage.uniform_filter(Ix * Iy, block, mode="mirror")
    return Sxx * Syy - Sxy**2 - k * (Sxx + Syy) ** 2


def nms_peaks(R, radius=3, rel=0.01):
    peaks = (R == ndimage.maximum_filter(R, size=2 * radius + 1)) & (R > rel * R.max())
    return np.argwhere(peaks)  # (row, col)


# ------------------------------------------------------------------ TTC
def ttc_camera(kp_prev, kp_curr, dt, min_dist=40.0):
    """Median ratio of keypoint-pair distances: h1/h0 = d0/d1  =>  TTC = -dt / (1 - ratio)."""
    i, j = np.triu_indices(len(kp_curr), 1)
    d_curr = np.linalg.norm(kp_curr[i] - kp_curr[j], axis=1)
    d_prev = np.linalg.norm(kp_prev[i] - kp_prev[j], axis=1)
    ok = (d_prev > 1e-6) & (d_curr >= min_dist)
    ratios = d_curr[ok] / d_prev[ok]
    return -dt / (1 - np.median(ratios)), ratios


def ttc_lidar(x_prev, x_curr, dt, robust):
    d0, d1 = (np.median(x_prev), np.median(x_curr)) if robust else (x_prev.min(), x_curr.min())
    return d1 * dt / (d0 - d1)


# ------------------------------------------------------------------ YOLO-style post-processing
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def nms(boxes, scores, iou_thr=0.4):
    order, keep = list(np.argsort(scores)[::-1]), []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [j for j in order if iou(boxes[best], boxes[j]) <= iou_thr]
    return keep


def main():
    rng = np.random.default_rng(3)
    os.makedirs("results", exist_ok=True)
    tex, bg = car_texture(rng), background()
    v, dt, d0, n = 5.0, 0.1, 15.0, 10
    dists = d0 - v * dt * np.arange(n)
    frames, boxes, scans = [], [], []
    for d in dists:
        img, box = render(tex, bg, d, rng)
        frames.append(img)
        boxes.append(box)
        scans.append(lidar_rear_face(d, rng))

    # 1. SIFT + ratio test -> camera TTC; lidar TTC with min vs median
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2)
    feats = []
    for img, box in zip(frames, boxes):
        mask = np.zeros_like(img)
        x0, y0, x1, y1 = box.astype(int)
        mask[y0:y1, x0:x1] = 255
        feats.append(sift.detectAndCompute(img, mask))
    rows, ratio_example = [], None
    for k in range(1, n):
        (kp0, de0), (kp1, de1) = feats[k - 1], feats[k]
        good = [m for m, s in bf.knnMatch(de0, de1, k=2) if m.distance < 0.8 * s.distance]
        p0 = np.float32([kp0[m.queryIdx].pt for m in good])
        p1 = np.float32([kp1[m.trainIdx].pt for m in good])
        ok = inside(p0, shrink(boxes[k - 1], 0.05)) & inside(p1, shrink(boxes[k], 0.05))
        ttc_c, ratios = ttc_camera(p0[ok], p1[ok], dt)
        lid = [s[inside(project(s), shrink(b, 0.10))][:, 0] for s, b in ((scans[k - 1], boxes[k - 1]), (scans[k], boxes[k]))]
        rows.append((k, dists[k] / v, ttc_lidar(*lid, dt, False), ttc_lidar(*lid, dt, True), ttc_c, int(ok.sum())))
        if k == n - 1:
            ratio_example, match_pts = ratios, p1[ok]
    print("frame  TTC true  lidar(min)  lidar(median)  camera(SIFT)  matches")
    for k, t, lmin, lmed, cam, m in rows:
        print(f"{k:5d} {t:8.2f}s {lmin:10.2f}s {lmed:13.2f}s {cam:12.2f}s {m:8d}")
    rows = np.array(rows)
    err = np.abs(rows[:, 2:5] - rows[:, 1:2]).mean(0)
    print(f"mean |error|  lidar(min) {err[0]:.3f} s | lidar(median) {err[1]:.3f} s | camera {err[2]:.3f} s")

    # 2. Harris from scratch vs OpenCV on the last frame's car crop
    x0, y0, x1, y1 = (boxes[-1] + np.array([-10, -10, 10, 10])).astype(int)
    crop = frames[-1][y0:y1, x0:x1]
    R_own = harris_response(crop)
    R_cv = cv2.cornerHarris(crop.astype(np.float32), blockSize=3, ksize=3, k=0.04)
    own, ref = nms_peaks(R_own), nms_peaks(R_cv.astype(np.float64))
    diff = np.abs(R_own / R_own.max() - R_cv / R_cv.max()).max()
    same = len(set(map(tuple, own)) & set(map(tuple, ref)))
    print(f"Harris  own {len(own)} corners | OpenCV {len(ref)} | identical {same} | max normalised diff {diff:.1e}")

    # 3. YOLO-style post-processing on synthetic detector output
    truth = np.array([[560, 170, 700, 290], [880, 180, 915, 260]], float)  # car, pedestrian
    cand, score = [], []
    for tb, base in zip(truth, (0.92, 0.81)):
        for _ in range(6):  # overlapping duplicates around each object
            cand.append(tb + rng.normal(0, 6, 4))
            score.append(base - rng.uniform(0, 0.35))
    for _ in range(4):  # low-confidence false positives
        xy = rng.uniform([100, 120], [1100, 300])
        cand.append(np.r_[xy, xy + rng.uniform(30, 90, 2)])
        score.append(rng.uniform(0.05, 0.45))
    cand, score = np.array(cand), np.array(score)
    conf = np.flatnonzero(score >= 0.5)  # confidence threshold first, then NMS
    keep = conf[nms(cand[conf], score[conf], 0.4)]
    print(f"YOLO post-processing  {len(cand)} candidates -> {len(conf)} above 0.5 -> {len(keep)} after NMS; "
          f"IoU with truth {[round(float(max(iou(cand[i], t) for t in truth)), 2) for i in keep]}")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
    bx0, by0, bx1, by1 = (boxes[-1] + np.array([-40, -30, 40, 30])).astype(int)
    a = ax[0, 0]
    a.imshow(frames[-1][by0:by1, bx0:bx1], cmap="gray", vmin=0, vmax=255)
    uv = project(scans[-1]) - [bx0, by0]
    sc = a.scatter(uv[:, 0], uv[:, 1], c=scans[-1][:, 0], cmap="jet_r", s=9, vmin=dists[-1] - 0.8, vmax=dists[-1] + 0.1)
    a.scatter(*(match_pts - [bx0, by0]).T, s=14, facecolors="none", edgecolors="lime", lw=0.8)
    sb = shrink(boxes[-1], 0.10) - [bx0, by0, bx0, by0]
    a.add_patch(Rectangle(sb[:2], sb[2] - sb[0], sb[3] - sb[1], fill=False, ec="cyan", lw=1.2))
    fig.colorbar(sc, ax=a, fraction=0.046, label="lidar x [m]")
    a.set_title("(a) Lidar projected into the image + SIFT matches")
    a.axis("off")
    blur = cv2.GaussianBlur(crop, (0, 0), 1.5)
    gmag = np.hypot(cv2.Sobel(blur, cv2.CV_64F, 1, 0), cv2.Sobel(blur, cv2.CV_64F, 0, 1))
    ax[0, 1].imshow(gmag, cmap="magma")
    ax[0, 1].set_title("(b) Gaussian blur (sigma 1.5) + Sobel |gradient|")
    ax[0, 1].axis("off")
    ax[0, 2].imshow(crop, cmap="gray")
    ax[0, 2].scatter(own[:, 1], own[:, 0], s=40, facecolors="none", edgecolors="yellow", label="own Harris + NMS")
    ax[0, 2].scatter(ref[:, 1], ref[:, 0], s=8, c="red", marker="x", label="cv2.cornerHarris")
    ax[0, 2].legend(loc="lower right", fontsize=8)
    ax[0, 2].set_title(f"(c) Harris: {same}/{len(ref)} corners identical")
    ax[0, 2].axis("off")
    a = ax[1, 0]
    a.plot(rows[:, 0], rows[:, 1], "k-", lw=2, label="truth d/v")
    a.plot(rows[:, 0], rows[:, 2], "r.--", label="lidar, min x (outlier-prone)")
    a.plot(rows[:, 0], rows[:, 3], "bs-", ms=4, label="lidar, median x")
    a.plot(rows[:, 0], rows[:, 4], "g^-", ms=5, label="camera, median distance ratio")
    a.set(xlabel="frame", ylabel="TTC [s]", title="(d) Time to collision per frame")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a = ax[1, 1]
    lo, hi = np.percentile(ratio_example, [2, 98])  # mismatched pairs create a long tail
    a.hist(ratio_example, bins=60, range=(lo, hi), color="tab:green", alpha=0.8)
    med = np.median(ratio_example)
    a.axvline(med, color="k", lw=2, label=f"median {med:.4f} -> TTC {-dt / (1 - med):.2f} s")
    a.axvline(dists[-2] / dists[-1], color="r", ls="--", label=f"true d0/d1 {dists[-2] / dists[-1]:.4f}")
    a.set(xlabel="distance ratio  d_curr / d_prev", ylabel="keypoint pairs", title="(e) Distance ratios, last frame pair")
    a.legend(fontsize=8)
    a = ax[1, 2]
    a.set(xlim=(80, 1160), ylim=(330, 100), title="(f) YOLO post-processing: threshold 0.5 + NMS 0.4")
    for i, b in enumerate(cand):
        kept = i in keep
        a.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, lw=2.2 if kept else 0.7,
                              ec="tab:blue" if kept else ("0.6" if score[i] < 0.5 else "tab:orange")))
    for t in truth:
        a.add_patch(Rectangle(t[:2], t[2] - t[0], t[3] - t[1], fill=False, ls=":", ec="k"))
    a.plot([], [], "0.6", label="below confidence")
    a.plot([], [], "tab:orange", label="suppressed by NMS")
    a.plot([], [], "tab:blue", lw=2.2, label="kept")
    a.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("results/camera_sim.png", dpi=105)
    print("-> results/camera_sim.png")


if __name__ == "__main__":
    main()
