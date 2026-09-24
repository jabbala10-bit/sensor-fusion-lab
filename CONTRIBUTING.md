# Contributing

This is a teaching lab, so clarity of the algorithm beats cleverness of the implementation.
Two rules matter more than style: every result must be reproducible, and the handbook must stay
true to the code.

## Ground rules

- **Write the algorithm out.** RANSAC, the KD-tree, the FFT, CFAR and the unscented transform are
  implemented from scratch on purpose. Reach for a library only when the point being taught is the
  library itself (PCL and OpenCV appear in the handbook as reference code).
- **Keep dependencies optional.** The lidar and radar labs must build with nothing installed.
  New dependencies belong behind a `find_package` guard in `cpp/CMakeLists.txt`.
- **Seed every random draw.** `std::mt19937 rng(seed)` and `np.random.default_rng(seed)`, never the
  global numpy functions, so the numbers in the README reproduce exactly.
- **Simulate the sensor.** No lab may require a downloaded dataset or network weights.
- **Report failures, do not hide them.** The false alarm in the radar output and the split truck in
  the lidar output are the lesson; keep that kind of honesty in new labs.

## Build, run, check

```bash
make cpp          # configure, build and run the C++ labs
make py           # run the Python simulations, writing python/results/
make lint         # ruff for Python, clang-format check for C++
make clean
```

CI runs the same steps on ubuntu-24.04 for both toolchains and uploads the figures.

## Style

- C++17, 2-space indent, `.clang-format` for new files. Existing sources use aligned trailing
  comments that a reformat may disturb, so format only what you touch.
- Python: ruff with a 135 column limit (`ruff.toml`), gating E, F, W, I and UP; `ruff check --select B` adds advisory suggestions. The one silenced rule is `E741` for `I` as
  image intensity in `camera_sim.py`, which matches the Harris formulation.
- Prefer a comment that gives the reason over one that restates the code: why a guard cell is
  skipped, not that a loop increments.

## When behaviour changes

1. Rerun the affected lab and update the numbers in `README.md` under "Results to expect".
2. Update the matching handbook tab: several sections quote these sources verbatim and print their
   actual output, so renaming a function or changing a default makes the documentation wrong.
3. Add a line to `CHANGELOG.md`.

## Commit messages

Conventional commits, scoped by section: `feat(radar): add OS-CFAR variant`,
`fix(kalman): predict to the measurement timestamp`, `docs: update TTC results`.
