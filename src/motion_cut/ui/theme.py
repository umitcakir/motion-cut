"""Central colour tokens and stylesheets for the Motion Cut UI."""

from __future__ import annotations

from string import Template

# Surfaces, recessed to raised. Neutral slate so the camera feed and accents pop.
INSET = "#010409"
BG = "#0d1117"
SURFACE = "#161b22"
SURFACE_RAISED = "#21262d"
SURFACE_HOVER = "#30363d"

BORDER = "#30363d"
BORDER_STRONG = "#3d444d"

TEXT = "#e6edf3"
TEXT_MUTED = "#9198a1"
TEXT_DIM = "#6e7681"

ACCENT = "#1f6feb"
ACCENT_HOVER = "#388bfd"
ACCENT_TEXT = "#ffffff"
ACCENT_SOFT_BG = "#0d2a5e"
ACCENT_SOFT_TEXT = "#cae0ff"

SUCCESS = "#3fb950"
SUCCESS_SOFT_BG = "#0f2417"
SUCCESS_BORDER = "#238636"

WARNING = "#e3b341"
WARNING_BORDER = "#9e6a03"
WARNING_SOFT_BG = "#2b2007"
WARNING_HOVER_BG = "#3a2c09"

DANGER = "#f85149"
DANGER_BORDER = "#8e2a26"
DANGER_SOFT_BG = "#2d1214"
DANGER_HOVER_BG = "#3d181a"

INFO = "#58a6ff"
SYSTEM = "#a371f7"

DISABLED_BG = "#1c2128"
DISABLED_TEXT = "#6e7681"

# Hand landmark rendering.
LANDMARK_BONE = "#768390"
LANDMARK_TIP = "#ffffff"
LANDMARK_MID = SUCCESS
LANDMARK_BASE = INFO

_TOKENS = {
    name: value
    for name, value in list(globals().items())
    if name.isupper() and isinstance(value, str)
}

_SHARED = """
QWidget {
    background: $BG;
    color: $TEXT;
    font-size: 13px;
}

QToolTip {
    background: $SURFACE_RAISED;
    color: $TEXT;
    border: 1px solid $BORDER_STRONG;
    padding: 4px 6px;
}

QLabel, QCheckBox {
    background: transparent;
}

QPushButton {
    background: $SURFACE_RAISED;
    border: 1px solid $BORDER;
    border-radius: 8px;
    padding: 7px 12px;
}

QPushButton:hover { background: $SURFACE_HOVER; }
QPushButton:pressed { background: $BORDER; }

QPushButton:disabled {
    background: $DISABLED_BG;
    border: 1px solid $BORDER;
    color: $DISABLED_TEXT;
}

QPushButton#primaryButton {
    background: $ACCENT;
    border: 1px solid $ACCENT_HOVER;
    color: $ACCENT_TEXT;
    font-weight: 600;
}

QPushButton#primaryButton:hover { background: $ACCENT_HOVER; }

QPushButton#primaryButton:disabled {
    background: $DISABLED_BG;
    border: 1px solid $BORDER;
    color: $DISABLED_TEXT;
}

QPushButton#secondaryButton {
    background: $SURFACE_RAISED;
    border: 1px solid $BORDER_STRONG;
}

QPushButton#secondaryButton:hover { background: $SURFACE_HOVER; }

QPushButton#secondaryButton:disabled {
    background: $DISABLED_BG;
    border: 1px solid $BORDER;
    color: $DISABLED_TEXT;
}

QPushButton#warningButton {
    background: $WARNING_SOFT_BG;
    border: 1px solid $WARNING_BORDER;
    color: $WARNING;
    font-weight: 600;
}

QPushButton#warningButton:hover { background: $WARNING_HOVER_BG; }

QPushButton#warningButton:disabled {
    background: $DISABLED_BG;
    border: 1px solid $BORDER;
    color: $DISABLED_TEXT;
}

QPushButton#dangerButton {
    background: $DANGER_SOFT_BG;
    border: 1px solid $DANGER_BORDER;
    color: $DANGER;
    font-weight: 600;
}

QPushButton#dangerButton:hover { background: $DANGER_HOVER_BG; }

QPushButton#dangerButton:disabled {
    background: $DISABLED_BG;
    border: 1px solid $BORDER;
    color: $DISABLED_TEXT;
}

QLineEdit, QComboBox, QKeySequenceEdit {
    background: $BG;
    border: 1px solid $BORDER;
    border-radius: 8px;
    padding: 6px 10px;
    selection-background-color: $ACCENT;
    selection-color: $ACCENT_TEXT;
}

QLineEdit:focus, QComboBox:focus, QKeySequenceEdit:focus {
    border: 1px solid $ACCENT;
}

QComboBox::drop-down { border: none; width: 18px; }

QComboBox QAbstractItemView {
    background: $SURFACE;
    border: 1px solid $BORDER;
    selection-background-color: $ACCENT_SOFT_BG;
    selection-color: $TEXT;
    outline: none;
}

QCheckBox { spacing: 7px; }

QSlider::groove:horizontal {
    background: $SURFACE_RAISED;
    border: 1px solid $BORDER;
    height: 8px;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: $ACCENT_HOVER;
    border: 1px solid $ACCENT_HOVER;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::sub-page:horizontal { background: $ACCENT; border-radius: 4px; }
QSlider::add-page:horizontal { background: $BG; border-radius: 4px; }
"""

_MAIN = """
QGroupBox {
    border: 1px solid $BORDER;
    border-radius: 10px;
    padding: 10px;
    background: $SURFACE;
}

QLabel#statusLabel {
    background: $INSET;
    border: 1px solid $BORDER;
    border-radius: 8px;
    padding: 7px 9px;
}

QLabel#hintLabel { color: $TEXT_MUTED; }

QGroupBox#gestureManagerBox {
    border: 1px solid $BORDER_STRONG;
    background: $SURFACE;
    margin-top: 10px;
    padding-top: 16px;
}

QGroupBox#gestureManagerBox::title,
QGroupBox#gestureSequenceBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: $TEXT_MUTED;
    font-weight: 600;
}

QFrame#gestureActionBox {
    background: $BG;
    border: 1px solid $BORDER;
    border-radius: 8px;
}

QGroupBox#gestureSequenceBox {
    border: 1px solid $BORDER;
    background: $BG;
    margin-top: 10px;
    padding-top: 16px;
}

QLabel#sectionTitle {
    color: $INFO;
    font-weight: 700;
}

QLabel#selectedGestureLabel {
    background: $ACCENT_SOFT_BG;
    border: 1px solid $ACCENT;
    border-radius: 6px;
    color: $ACCENT_SOFT_TEXT;
    font-size: 14px;
    font-weight: 700;
    padding: 8px 10px;
}

QLabel#previewLabel {
    background: $INSET;
    color: $TEXT_DIM;
    border: 1px solid $BORDER;
    border-radius: 10px;
}

QPushButton#recordButton {
    background: $ACCENT_SOFT_BG;
    border: 1px solid $ACCENT;
    color: $ACCENT_SOFT_TEXT;
    font-weight: 700;
    font-size: 13px;
}

QPushButton#recordButton:hover {
    background: $ACCENT;
    border: 1px solid $ACCENT_HOVER;
    color: $ACCENT_TEXT;
}

QScrollBar:vertical {
    background: $BG;
    width: 10px;
    margin: 2px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: $BORDER_STRONG;
    min-height: 28px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover { background: $TEXT_DIM; }

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
    border: none;
    background: transparent;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}

QListWidget {
    background: $BG;
    border: 1px solid $BORDER;
    border-radius: 8px;
    padding: 4px;
}

QListWidget::item {
    padding: 4px 6px;
    border-radius: 6px;
}

QListWidget::item:selected {
    background: $ACCENT_SOFT_BG;
    color: $TEXT;
}
"""

_WIZARD = """
QLabel#stepDotActive {
    background: $ACCENT;
    border: 1px solid $ACCENT_HOVER;
    border-radius: 6px;
    color: $ACCENT_TEXT;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 2px;
}

QLabel#stepDotDone {
    background: $SUCCESS_SOFT_BG;
    border: 1px solid $SUCCESS_BORDER;
    border-radius: 6px;
    color: $SUCCESS;
    font-size: 11px;
    padding: 4px 2px;
}

QLabel#stepDotInactive {
    background: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: 6px;
    color: $TEXT_DIM;
    font-size: 11px;
    padding: 4px 2px;
}

QLabel#descLabel {
    color: $TEXT;
    font-size: 13px;
    padding: 2px 0;
}

QLabel#statusHint {
    color: $TEXT_MUTED;
    font-size: 12px;
}

QLabel#errorLabel {
    color: $DANGER;
    font-size: 12px;
}

QLabel#nameOk {
    color: $SUCCESS;
    font-size: 12px;
}

QLabel#detectionIdle {
    background: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: 8px;
    color: $TEXT_MUTED;
    padding: 4px 10px;
    font-size: 12px;
}

QLabel#detectionArmed {
    background: $WARNING_SOFT_BG;
    border: 1px solid $WARNING_BORDER;
    border-radius: 8px;
    color: $WARNING;
    padding: 4px 10px;
    font-size: 12px;
}

QLabel#detectionHit {
    background: $SUCCESS_SOFT_BG;
    border: 1px solid $SUCCESS_BORDER;
    border-radius: 8px;
    color: $SUCCESS;
    font-weight: 700;
    padding: 4px 10px;
    font-size: 13px;
}

QPushButton#captureButton {
    background: $ACCENT_SOFT_BG;
    border: 1px solid $ACCENT;
    color: $ACCENT_SOFT_TEXT;
    font-weight: 600;
    font-size: 13px;
}

QPushButton#captureButton:hover {
    background: $ACCENT;
    color: $ACCENT_TEXT;
}

QPushButton#captureButton:disabled {
    background: $DISABLED_BG;
    border: 1px solid $BORDER;
    color: $DISABLED_TEXT;
}

QPushButton#cancelButton {
    background: $SURFACE_RAISED;
    color: $TEXT_MUTED;
}
"""


def main_window_stylesheet() -> str:
    return Template(_SHARED + _MAIN).substitute(_TOKENS)


def wizard_stylesheet() -> str:
    return Template(_SHARED + _WIZARD).substitute(_TOKENS)


def panel_stylesheet() -> str:
    """Frame around the scrollable control column."""
    return (
        f"QScrollArea {{ border: 1px solid {BORDER}; border-radius: 10px; background: {BG}; }}"
    )


def snapshot_stylesheet() -> str:
    """Inset box that holds a captured pose thumbnail."""
    return (
        f"background: {INSET}; border: 1px solid {BORDER};"
        f" border-radius: 8px; color: {TEXT_MUTED};"
    )
