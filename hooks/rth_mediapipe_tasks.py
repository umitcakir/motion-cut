# Runtime hook: fix mediapipe.tasks imports when running frozen under PyInstaller.
import sys
import os
import pathlib
import types

# ── 1. Stub out matplotlib so mediapipe.tasks.python.vision.drawing_utils loads ──
# drawing_utils imports matplotlib.pyplot at the top level; we never call those
# drawing helpers, so a minimal stub is enough.
_mpl = types.ModuleType("matplotlib")
_plt = types.ModuleType("matplotlib.pyplot")
_mpl.pyplot = _plt
sys.modules.setdefault("matplotlib", _mpl)
sys.modules.setdefault("matplotlib.pyplot", _plt)

# ── 2. Fix importlib.resources.files('mediapipe.tasks.c') in frozen bundles ──
if hasattr(sys, '_MEIPASS'):
    _lib_dir = pathlib.Path(sys._MEIPASS) / 'mediapipe' / 'tasks' / 'c'

    if _lib_dir.is_dir():
        from importlib import resources as _importlib_resources

        _orig_files = _importlib_resources.files

        def _patched_files(package):
            pkg_name = getattr(package, '__name__', str(package))
            if pkg_name == 'mediapipe.tasks.c':
                return _lib_dir
            return _orig_files(package)

        _importlib_resources.files = _patched_files
