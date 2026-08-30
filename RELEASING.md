# Releasing Motion Cut

## Before Tagging

1. Confirm the working tree contains only intended changes.
2. Run the full test suite:

   ```sh
   PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
   ```

3. Update the version in `pyproject.toml` and document user-visible changes in the GitHub release notes.
4. Verify the README platform instructions and known limitations are still accurate.
5. Confirm no local database, downloaded model, build output, credentials, or personal recordings are staged.

## Create The Release

The GitHub Actions workflow builds native artifacts on Linux, Windows, and macOS when a version tag is pushed.

```sh
git tag -a v0.1.0 -m "Motion Cut v0.1.0"
git push origin v0.1.0
```

Wait for the **Build Executables** workflow to finish. It uploads these release assets:

- `motion-cut-linux-x86_64`
- `motion-cut-windows-x86_64.exe`
- `motion-cut-macos-x86_64.tar.gz`

The release is created automatically for version tags. Publishing a GitHub Release that targets `main` also runs the build and attaches the same assets to that release. Before announcing it, download and smoke-test each artifact on its target platform when possible.

## Signing

Current artifacts are unsigned. Do not describe them as signed, notarized, or malware-scanned. Windows users may see SmartScreen warnings and macOS users may need Control-click > Open. Add code-signing and macOS notarization before presenting the binaries as production-ready.
