# Changelog

All notable changes to this lab. Format follows Keep a Changelog; versions follow semantic
versioning.

## [0.1.0] - 2026-09-24

First release: nine programs, 2418 lines, verified by a clean build and a full run.

### Added

- **Lidar** (`cpp/lidar/lidar_pipeline.cpp`, `python/lidar_sim.py`): ray-cast lidar simulator,
  voxel grid, ROI and ego-roof crop, RANSAC ground plane with adaptive iteration count and
  least-squares refit, KD-tree, Euclidean clustering with fixed and range-adaptive tolerance,
  AABB / PCA / minimum-area / L-shape box fitting, range-image view.
- **Camera** (`cpp/camera/camera_ttc.cpp`, `python/camera_sim.py`): synthetic approach sequence,
  pinhole and lidar-to-image projection, Harris with non-maximum suppression, seven detectors
  against six descriptors with ratio-test matching, camera and lidar time to collision, IoU and
  non-maximum suppression for detector post-processing.
- **Radar** (`cpp/radar/fmcw_radar.cpp`, `python/radar_sim.py`, `python/radar_mtt_sim.py`):
  waveform design from requirements, beat signal on an eight-element array, own radix-2 FFT,
  range-Doppler map, 2D CA-CFAR, angle FFT, rain-clutter comparison against a fixed threshold,
  detection clustering and a GNN tracker with gating, Hungarian assignment and M-of-N confirmation.
- **Kalman** (`cpp/kalman/fusion_filters.cpp`, `python/kalman_sim.py`): CTRV trajectory generator,
  KF on lidar only, EKF fusing lidar and radar, UKF with the CTRV model, RMSE, NIS consistency and
  a process-noise tuning sweep.
- Repo metadata: MIT license, citation file, CI for both toolchains, contributing guide, editor and
  formatter configuration, Makefile shortcuts.
