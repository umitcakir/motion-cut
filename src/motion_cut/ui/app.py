from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def launch_app() -> int:
    app = QApplication(sys.argv)
    from motion_cut.ui.window import MainWindow
    window = MainWindow()
    app.setWindowIcon(window.windowIcon())
    window.show()
    QTimer.singleShot(0, window.start_camera)
    return app.exec()
