# Repository Guidelines

## Project Structure & Module Organization
- `performance_test/`: C++ ROS 2 benchmarking core; production code in `src/`, public headers in `include/`, configurable transports in `plugins/`, and automation helpers in `helper_scripts/`.
- `performance_report/`: Python orchestration and visualization package; CLI entry points live in `performance_report/`, experiment YAMLs in `cfg/`, and tests under `test/`.
- `plotter/` and `performance_report/cfg/*`: reference plotting scripts and example report pipelines—update together when log schemas change.
- `dockerfiles/`, `third_party/`, `patches/`: curated build environments and vendored dependencies mirrored by Bazel `MODULE.bazel` and `repositories.bzl`; keep these in sync when bumping middleware or Python packages.

## Build, Test, and Development Commands
- Colcon build (default): `colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DPERFORMANCE_TEST_PLUGIN=<PLUGIN>`; source `install/setup.bash` before running binaries.
- Bazel build parity: `bazel build //performance_test:performance_test` and `bazel build //performance_report:runner` to match CI toolchains.
- Benchmark example: `ros2 run performance_test perf_test --communication CycloneDDS --msg Array1k --rate 100 --max-runtime 30 --logfile experiment/log.csv`.
- Report pipeline: `ros2 run performance_report runner --log-dir perf_logs --configs performance_report/cfg/runner/run_one_experiment.yaml`, then `ros2 run performance_report plotter --log-dir perf_logs --configs performance_report/cfg/plotter/plot_one_experiment.yaml`.

## Coding Style & Naming Conventions
- C++ targets C++17 with two-space indentation, braces on new lines, `CamelCase` for types, and lower_snake_case for functions and variables; group includes as standard, third-party, project.
- Plugin folders follow `plugins/<middleware>/`; expose new plugins through `performance_test_with_plugin` in `performance_test/BUILD.bazel` and document flags in `README.md`.
- Python adheres to `flake8` and `pydocstyle` (enforced via `pytest` suite); prefer black-compatible formatting and explicit type hints for new modules.

## Testing Guidelines
- C++ unit tests use GoogleTest in `performance_test/test/src`; run `colcon test --packages-select performance_test` after any library or CLI change.
- Python utilities validate via `pytest -q performance_report/test`; ensure new plotting or reporting features add fixtures/configs under `performance_report/cfg/`.
- For multi-experiment workflows, stash generated logs in `perf_logs/` (ignored by git) and include reproduction commands in PR descriptions.

## Commit & Pull Request Guidelines
- Recent history favors concise, sentence-case subjects (e.g., “Add CI jobs for Kilted”); keep summaries under 72 characters and use imperative tone where possible.
- Reference issues or ROS 2 tickets in the body (`Fixes #123`) and call out targeted distros, middleware plugins, or dependency revisions.
- PRs should outline testing (`colcon build`, `colcon test`, `pytest`) and attach relevant artifacts—ASCII log excerpts, plotted PNGs, or `perf_logs` directories zipped as releases.

## Security & Configuration Tips
- Run `helper_scripts/security_setup.bash` when enabling secure DDS transports; exclude generated certificates from commits.
- Update `third_party/python/requirements.txt` with `bazel run //third_party/python:requirements.update` and mirror changes in the Dockerfiles to keep developer images reproducible.
