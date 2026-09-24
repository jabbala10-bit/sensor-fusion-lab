#!/usr/bin/env python3
"""FMCW radar simulation: waveform design, range-Doppler map, rain clutter,
fixed threshold vs 2D CA-CFAR, clustering of detected cells, angle of arrival.
Writes results/radar_sim.png."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

C, FC = 3e8, 77e9
LAM = C / FC
RANGE_RES, R_MAX = 1.0, 200.0
B = C / (2 * RANGE_RES)              # range resolution = c / 2B
TC = 5.5 * 2 * R_MAX / C             # chirp time: 5.5 x round trip at max range
S = B / TC                           # slope [Hz/s]
NR, ND, NA = 256, 128, 8             # IQ samples per chirp, chirps per frame, receive antennas
FS = NR / TC
V_RES, V_MAX = LAM / (2 * ND * TC), LAM / (4 * TC)

TARGETS = [  # name, range [m], range rate [m/s] (+ away), angle [deg], RCS [dBsm]
    ("car ahead", 110, -20.0, 2, 10),
    ("car, receding", 60, 8.0, -15, 10),
    ("pedestrian", 45, -1.5, 20, -5),
    ("truck", 160, -5.0, -5, 20),
]


def scatterer(amp, r, v, ang_deg, rng):
    """Separable beat-signal model: range tone + Doppler phase ramp + array phase (d = lambda/2)."""
    fb = S * 2 * r / C + 2 * v / LAM                          # beat frequency incl. Doppler coupling
    fast = np.exp(2j * np.pi * fb * np.arange(NR) / FS)
    slow = np.exp(2j * np.pi * (2 * v / LAM) * np.arange(ND) * TC)
    array = np.exp(1j * np.pi * np.arange(NA) * np.sin(np.deg2rad(ang_deg)))
    return amp * np.exp(2j * np.pi * rng.random()) * array[:, None, None] * slow[None, :, None] * fast[None, None, :]


def simulate(rng, rain=True):
    amp_ref = 10 ** (-20 / 20) * 110**2 / 10 ** (10 / 20)      # car at 110 m: -20 dB per-sample SNR
    cube = (rng.normal(size=(NA, ND, NR)) + 1j * rng.normal(size=(NA, ND, NR))) / np.sqrt(2)
    for _, r, v, ang, rcs in TARGETS:
        cube += scatterer(amp_ref * 10 ** (rcs / 20) / r**2, r, v, ang, rng)
    if rain:  # many weak drops at short range with a spread of radial speeds
        for _ in range(400):
            r = rng.uniform(3, 35)
            cube += scatterer(amp_ref * 10 ** (-24 / 20) / r**2, r, rng.normal(0, 1.5), rng.uniform(-40, 40), rng)
    return cube


def range_doppler(cube):
    x = cube * np.hanning(NR)[None, None, :]
    x = np.fft.fft(x, axis=2)                                  # fast time -> range
    x = x * np.hanning(ND)[None, :, None]
    return np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)     # slow time -> Doppler


def ca_cfar_2d(p, tr=8, td=4, gr=2, gd=2, pfa=1e-6):
    """Cell-averaging CFAR on linear power; returns detection mask, threshold and alpha."""
    big = np.ones((2 * (td + gd) + 1, 2 * (tr + gr) + 1))
    guard = np.zeros_like(big)
    guard[td:td + 2 * gd + 1, tr:tr + 2 * gr + 1] = 1
    kernel = big - guard                                       # training ring, guard cells + CUT excluded
    n_train = int(kernel.sum())
    alpha = n_train * (pfa ** (-1 / n_train) - 1)
    thr = alpha * ndimage.correlate(p, kernel, mode="constant") / n_train
    valid = np.zeros_like(p, bool)
    valid[td + gd:-(td + gd), tr + gr:-(tr + gr)] = True       # skip edges without a full window
    return (p > thr) & valid, thr, alpha, n_train


def main():
    rng = np.random.default_rng(11)
    os.makedirs("results", exist_ok=True)
    print(f"B {B / 1e6:.0f} MHz | Tc {TC * 1e6:.2f} us | slope {S / 1e12:.2f} MHz/us | Fs {FS / 1e6:.1f} MHz")
    print(f"range res {C / (2 * B):.2f} m, max {FS * C / (2 * S):.0f} m | velocity res {V_RES:.2f} m/s, "
          f"max +-{V_MAX:.1f} m/s | angle res ~{np.degrees(2 / NA):.1f} deg")
    X = range_doppler(simulate(rng))
    p = np.abs(X[0]) ** 2                                      # antenna-0 range-Doppler power map
    ranges = np.arange(NR) * C / (2 * B)
    vels = (np.arange(ND) - ND // 2) * V_RES
    rain_zone = (ranges[None, :] < 38) & (np.abs(vels[:, None]) < 6)
    noise = np.median(p[~rain_zone]) / np.log(2)               # median of exponential = mean ln 2

    det, thr, alpha, n_train = ca_cfar_2d(p)
    fixed_db = 17.0                                            # just low enough for the weakest target
    fixed = p > noise * 10 ** (fixed_db / 10)
    print(f"CA-CFAR {n_train} training cells, alpha {alpha:.1f} ({10 * np.log10(alpha):.1f} dB) | "
          f"fixed threshold {fixed_db:.0f} dB over noise")
    print(f"cells detected  fixed: {fixed.sum()} ({(fixed & rain_zone).sum()} in rain) | "
          f"CA-CFAR: {det.sum()} ({(det & rain_zone).sum()} in rain)")

    labels, n = ndimage.label(det, structure=np.ones((3, 3)))  # 8-connected clusters of cells
    print(f"\n{'cluster':>7} {'range m':>8} {'vel m/s':>8} {'angle':>6} {'SNR dB':>6}  match")
    found, angle_spectra = set(), []
    for k in range(1, n + 1):
        cells = np.argwhere(labels == k)
        w = p[labels == k]
        r_est = (cells[:, 1] * w).sum() / w.sum() * RANGE_RES
        v_est = ((cells[:, 0] - ND // 2) * w).sum() / w.sum() * V_RES
        d_pk, r_pk = cells[np.argmax(w)]
        spec = np.fft.fftshift(np.fft.fft(X[:, d_pk, r_pk], 64))  # angle FFT, zero-padded
        k_pk = np.argmax(np.abs(spec)) - 32
        ang = np.degrees(np.arcsin(np.clip(2 * k_pk / 64, -1, 1)))
        match = [t for t in TARGETS if abs(t[1] - r_est) < 3 and abs(t[2] - v_est) < 3]
        name = match[0][0] if match else ("rain clutter" if rain_zone[d_pk, r_pk] else "false alarm")
        if match:
            found.add(name)
            angle_spectra.append((name, spec))
        print(f"{k:7d} {r_est:8.2f} {v_est:8.2f} {ang:6.1f} {10 * np.log10(p[d_pk, r_pk] / noise):6.1f}  {name}")
    print(f"targets detected {len(found)}/{len(TARGETS)}")

    # ------------------------------------------------------------------ figure
    db = lambda a: 10 * np.log10(a / noise)  # noqa: E731
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.6))
    prof = np.abs(np.fft.fft(simulate(np.random.default_rng(5), rain=False)[0, 0] * np.hanning(NR))) ** 2
    ax[0, 0].plot(ranges, 10 * np.log10(prof / np.median(prof)), lw=0.8)
    for name, r, *_ in TARGETS:
        ax[0, 0].axvline(r, color="r", ls=":", lw=1)
    ax[0, 0].set(xlabel="range [m]", ylabel="power [dB]", title="(a) Range FFT of one chirp (targets dotted)")
    im = ax[0, 1].imshow(db(p), aspect="auto", origin="lower", cmap="viridis", vmin=-5, vmax=35,
                         extent=[ranges[0], ranges[-1], vels[0], vels[-1]])
    dd, rr = np.nonzero(det)
    ax[0, 1].scatter(ranges[rr], vels[dd], s=6, c="r", label="CA-CFAR cells")
    ax[0, 1].set(xlabel="range [m]", ylabel="range rate [m/s]", title="(b) Range-Doppler map, antenna 0")
    ax[0, 1].legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax[0, 1], label="dB over noise")
    angles = np.degrees(np.arcsin(np.clip(2 * np.arange(-32, 32) / 64, -1, 1)))
    for name, spec in angle_spectra:
        ax[0, 2].plot(angles, 20 * np.log10(np.abs(spec) / np.abs(spec).max()), label=name)
    ax[0, 2].set(xlabel="angle [deg]", ylabel="dB", ylim=(-30, 2), title="(c) Angle FFT at each target (8 rx, zero-padded to 64)")
    ax[0, 2].legend(fontsize=8)
    d_car = ND // 2 + int(round(-20 / V_RES))
    ax[1, 0].plot(ranges, db(p[d_car]), lw=0.8, label="signal")
    ax[1, 0].plot(ranges, db(thr[d_car]), "r", label="CA-CFAR threshold")
    ax[1, 0].axhline(fixed_db, color="k", ls="--", label="fixed threshold")
    ax[1, 0].set(xlabel="range [m]", ylabel="dB over noise", ylim=(-15, 35), title="(d) Range cut at -20 m/s (car ahead)")
    ax[1, 0].legend(fontsize=8)
    r20 = int(round(20 / RANGE_RES))
    ax[1, 1].plot(vels, db(p[:, r20]), lw=0.8, label="signal at 20 m (rain)")
    ax[1, 1].plot(vels, db(thr[:, r20]), "r", label="CA-CFAR threshold")
    ax[1, 1].axhline(fixed_db, color="k", ls="--", label="fixed threshold")
    ax[1, 1].set(xlabel="range rate [m/s]", ylabel="dB over noise", ylim=(-15, 35), title="(e) Doppler cut through rain clutter")
    ax[1, 1].legend(fontsize=8)
    zoom = ranges < 70
    ax[1, 2].imshow(db(p[:, zoom]), aspect="auto", origin="lower", cmap="gray", vmin=-5, vmax=35,
                    extent=[0, ranges[zoom][-1], vels[0], vels[-1]])
    fd, fr = np.nonzero(fixed[:, zoom])
    cd, cr = np.nonzero(det[:, zoom])
    ax[1, 2].scatter(ranges[fr], vels[fd], s=8, c="orange", label=f"fixed: {fixed[:, zoom].sum()} cells")
    ax[1, 2].scatter(ranges[cr], vels[cd], s=8, c="cyan", label=f"CA-CFAR: {det[:, zoom].sum()} cells")
    ax[1, 2].set(xlabel="range [m]", ylabel="range rate [m/s]", ylim=(-15, 15), title="(f) Detections under 70 m: fixed vs CA-CFAR")
    ax[1, 2].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("results/radar_sim.png", dpi=105)
    print("-> results/radar_sim.png")


if __name__ == "__main__":
    main()
