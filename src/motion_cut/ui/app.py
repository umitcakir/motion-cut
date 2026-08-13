from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from motion_cut.ui.window import MainWindow


def launch_app() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.start_camera)
    return app.exec()
