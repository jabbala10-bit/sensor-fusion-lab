# Sensor Fusion Lab

[![ci](https://github.com/jabbala10-bit/sensor-fusion-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/jabbala10-bit/sensor-fusion-lab/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Runnable companion to the Sensor Fusion Mastery Handbook: four C++ programs and five Python
simulations covering lidar obstacle detection, camera geometry and time to collision, FMCW radar,
and Kalman filtering. Everything is written from scratch — RANSAC, KD-tree, FFT, CFAR, the
unscented transform — so each step is visible and hackable. No datasets are needed: every lab
simulates its own sensor.

## Layout

| Path | Section | What it does |
| --- | --- | --- |
| `cpp/lidar/lidar_pipeline.cpp` | Lidar | Ray-cast lidar simulator, voxel grid, ROI crop, RANSAC ground plane with least-squares refit, KD-tree, Euclidean clustering, bounding boxes |
| `cpp/camera/camera_ttc.cpp` | Camera | Synthetic approach sequence, Harris + NMS, 7 detectors x 6 descriptors, ratio-test matching, camera and lidar TTC |
| `cpp/radar/fmcw_radar.cpp` | Radar | Waveform design, beat signal on an 8-element array, own radix-2 FFT, range-Doppler map, 2D CA-CFAR, angle FFT |
| `cpp/kalman/fusion_filters.cpp` | Kalman | CTRV trajectory generator, KF (lidar only), EKF (lidar + radar), UKF with CTRV, RMSE and NIS |
| `python/lidar_sim.py` | Lidar | Same pipeline plus a rotated vehicle, range image, and AABB / PCA / min-area / L-shape boxes |
| `python/camera_sim.py` | Camera | Harris from scratch vs OpenCV, gradients, lidar-to-image projection, TTC, IoU and NMS |
| `python/radar_sim.py` | Radar | Range-Doppler map with rain clutter, fixed threshold vs CA-CFAR, angle spectra |
| `python/radar_mtt_sim.py` | Radar | Detection clustering and a GNN tracker with gating, Hungarian assignment and M-of-N confirmation |
| `python/kalman_sim.py` | Kalman | KF vs EKF vs UKF, NIS consistency, process-noise tuning sweep |

## Build and run

```bash
# C++ (needs a C++17 compiler; Eigen for the Kalman lab, OpenCV 4 for the camera lab)
sudo apt install libeigen3-dev libopencv-dev cmake      # Debian / Ubuntu
cd cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/lidar_pipeline            # optional argument: CSV file of labelled points
./build/camera_ttc                # or: ./build/camera_ttc FAST BRISK
./build/fmcw_radar
./build/fusion_filters
```

CMake skips a target whose dependency is missing, so the lidar and radar labs build with nothing
installed. BRIEF and FREAK descriptors compile only when OpenCV was built with opencv_contrib.

```bash
# Python (figures land in python/results/)
cd python && pip install -r requirements.txt
python lidar_sim.py && python camera_sim.py && python radar_sim.py
python radar_mtt_sim.py && python kalman_sim.py
```

## Results to expect

These are the numbers from the reference run on one CPU core; RNG seeds are fixed, so a rerun on
the same library versions reproduces them.

- **Lidar:** 24,538 points to 5,877 after filtering; RANSAC finds the ground in 5 adaptive
  iterations and recovers its height to 4 mm; a fixed 0.6 m cluster tolerance gives 10 clusters for
  6 objects, while a range-adaptive one gives exactly 6. In Python, L-shape fitting recovers the
  yaw of a car rotated 20 degrees with 0.0 degrees of error, against 18.3 for PCA.
- **Camera:** camera TTC within 0.008 s (SIFT) of truth; median-based lidar TTC within 0.025 s;
  the naive closest-point lidar TTC off by 1.03 s because of two ghost returns per frame. The
  from-scratch Harris matches `cv2.cornerHarris` on all 132 corners.
- **Radar:** all four targets found to within 0.06 m and 0.06 m/s; in rain a fixed threshold fires
  on 137 cells (108 of them rain) where CA-CFAR fires on 28 (2 in rain). The tracker holds 4.4
  confirmed tracks per scan against 4 true targets at 0.44 m RMSE, with 453 clutter detections.
- **Kalman:** velocity RMSE 0.60 m/s (lidar-only KF), 0.32 m/s (EKF fusion), 0.22 m/s (UKF with
  CTRV); UKF NIS above the 95% line in 7.6% of lidar and 3.6% of radar updates.

## Notes

- Every simulation is seeded. Change the seed to see how much of a result is luck: the radar lab's
  default seed happens to produce one CFAR false alarm, which is what a 10⁻⁶ false alarm rate over
  27,376 cells predicts every 37 frames or so.
- The PCL and YOLO snippets in the handbook are reference code, not part of this build: PCL is not
  a dependency here, and no network weights are shipped.
- Exercises for each section live in the handbook tab for that section.

## Project files

| File | Purpose |
| --- | --- |
| `Makefile` | `make cpp`, `make py`, `make lint`, `make clean` |
| `cpp/CMakeLists.txt` | Builds the four C++ labs, skipping any whose dependency is absent |
| `python/requirements.txt` | numpy, scipy, matplotlib, opencv-python |
| `ruff.toml`, `.clang-format`, `.editorconfig` | Lint and format settings used by CI and editors |
| `.github/workflows/ci.yml` | Builds and runs every lab on both toolchains, uploads the figures |
| `CONTRIBUTING.md` | Ground rules: reproducible seeds, optional dependencies, handbook stays in sync |
| `CHANGELOG.md` | Versioned history |
| `reference-output/` | Figures and images from the reference run, for comparison |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | MIT |
