# Contributing to Motion Cut

Thanks for helping improve Motion Cut. Small, focused contributions are easiest to review and safest for this alpha project.

## Before You Start

- Search existing issues before opening a new one.
- Open an issue first for behavior changes or larger features so the approach can be discussed.
- Do not include camera recordings, personal gesture databases, credentials, or machine-specific paths in issues or pull requests.

## Development Setup

```sh
git clone https://github.com/umitcakir/motion-cut.git
cd motion-cut
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Run the app with the launcher:

```sh
./start_motion_cut.sh
```

Run the test suite before opening a pull request:

```sh
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

## Pull Requests

- Keep each pull request focused on one problem.
- Include tests when changing gesture matching, persistence, camera handling, or shortcut dispatch.
- Update the README when user behavior, setup, permissions, or platform support changes.
- Describe the platform and desktop session used for manual testing.
- Do not commit `motion_cut.db`, downloaded models, build output, virtual environments, or secrets.
- By submitting a contribution, you agree that it is licensed under the [PolyForm Noncommercial 1.0.0 License](LICENSE).

## Reporting Compatibility Results

Camera and shortcut behavior can vary by OS, desktop session, and permissions. Useful reports include the Motion Cut version, operating system version, desktop session, camera model, permission state, and exact steps to reproduce the result.
