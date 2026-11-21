#!/usr/bin/env python3
"""Mario RL Player の起動スクリプト。

    .venv/bin/python run_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from mario_app import __version__
    from mario_app.ui import MainWindow
    from mario_app.ui.style import FALLBACK_FAMILY

    app = QApplication(sys.argv)
    app.setApplicationName("Mario RL Player")
    app.setApplicationDisplayName("Mario RL Player")
    app.setOrganizationName("sakai1250")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    app.setFont(QFont(FALLBACK_FAMILY, 13))
    try:
        # タイトルバー / メニューもダーク寄りにする（Qt 6.5+）
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except AttributeError:
        pass

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
