"""メニューバーと「について」ダイアログ。"""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMessageBox


def show_about(win) -> None:
    QMessageBox.about(
        win,
        "Mario RL Player",
        "<h3>Mario RL Player</h3>"
        "<p>DQN で Super Mario Bros. を学習・再生するアプリです。</p>"
        "<p>環境: gym-super-mario-bros / nes-py<br>学習・推論: PyTorch (CUDA / MPS / CPU)</p>",
    )


def build_menu(win) -> None:
    """``win``（MainWindow）にメニューを作り、``open_action`` / ``fullscreen_action`` を属性に持たせる。"""
    file_menu = win.menuBar().addMenu("ファイル")
    win.open_action = QAction("モデルを開く…", win)
    win.open_action.setShortcut(QKeySequence("Ctrl+O"))
    win.open_action.triggered.connect(lambda: win.model_box.choose_model())
    save_action = QAction("重みを保存", win)
    save_action.setShortcut(QKeySequence("Ctrl+S"))
    save_action.triggered.connect(win.save_weights)
    shot_action = QAction("スクリーンショットを保存", win)
    shot_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
    shot_action.triggered.connect(win.save_screenshot)
    quit_action = QAction("終了", win)
    quit_action.setShortcut(QKeySequence.Quit)
    quit_action.triggered.connect(win.close)
    for action in (win.open_action, save_action, shot_action):
        file_menu.addAction(action)
    file_menu.addSeparator()
    file_menu.addAction(quit_action)

    view_menu = win.menuBar().addMenu("表示")
    win.fullscreen_action = QAction("フルスクリーン", win, checkable=True)
    win.fullscreen_action.setShortcut(QKeySequence("Ctrl+Meta+F"))
    win.fullscreen_action.triggered.connect(win.toggle_fullscreen)
    view_menu.addAction(win.fullscreen_action)

    help_menu = win.menuBar().addMenu("ヘルプ")
    about = QAction("Mario RL Player について", win)
    about.triggered.connect(lambda: show_about(win))
    help_menu.addAction(about)
