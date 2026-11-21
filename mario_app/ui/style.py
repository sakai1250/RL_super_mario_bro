"""ファミコン調のパレット・フォント・Qt スタイルシート。"""

from __future__ import annotations

import logging
import sys

from PySide6.QtGui import QFont, QFontDatabase

from ..paths import FONT_DIR

logger = logging.getLogger(__name__)

PALETTE: dict[str, str] = {
    "body": "#E6DEC6",
    "panel": "#F3ECD8",
    "key": "#D8CFB4",
    "key_hi": "#F6F0DE",
    "key_lo": "#B3A783",
    "maroon": "#5C1B14",
    "gold": "#C9A24A",
    "ink": "#1F120E",
    "lcd": "#B9C29B",
    "lcd_ink": "#1E2418",
    "led_on": "#E5532D",
    "led_off": "#4B2A22",
    "cream": "#F1E7C8",
}

if sys.platform == "darwin":
    FALLBACK_FAMILY = "Hiragino Sans"
else:
    FALLBACK_FAMILY = "Noto Sans CJK JP"
_FONT_FILE = "DotGothic16-Regular.ttf"
_family: str | None = None


def load_pixel_font() -> str:
    """同梱のドットフォントを登録し、ファミリ名を返す。無ければ OS 標準フォント。"""
    global _family
    if _family is not None:
        return _family
    font_id = QFontDatabase.addApplicationFont(str(FONT_DIR / _FONT_FILE))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    if families:
        _family = families[0]
    else:
        logger.warning("同梱フォントを読み込めません。%s を使います", FALLBACK_FAMILY)
        _family = FALLBACK_FAMILY
    return _family


def pixel_font(pixel_size: int = 16) -> QFont:
    font = QFont(load_pixel_font())
    font.setPixelSize(pixel_size)
    font.setStyleStrategy(QFont.NoAntialias)
    return font


def build_qss(family: str) -> str:
    p = PALETTE
    return f"""
QWidget {{ color: {p['ink']}; font-family: "{family}", "{FALLBACK_FAMILY}", sans-serif; font-size: 16px; }}
QMainWindow, #Root {{ background: {p['body']}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: {p['key']}; width: 12px; border: 2px solid {p['ink']}; margin: 0; }}
QScrollBar::handle:vertical {{ background: {p['maroon']}; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QComboBox {{
    background: {p['lcd']}; color: {p['lcd_ink']}; border: 3px solid {p['ink']};
    border-radius: 0; padding: 2px 8px; min-height: 22px;
}}
QComboBox:disabled {{ color: {p['key_lo']}; }}
QComboBox::drop-down {{ border-left: 3px solid {p['ink']}; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {p['panel']}; color: {p['ink']}; border: 2px solid {p['ink']};
    selection-background-color: {p['maroon']}; selection-color: {p['cream']};
}}
QStatusBar {{ background: {p['lcd']}; color: {p['lcd_ink']}; border-top: 3px solid {p['ink']}; }}
QMenu {{ background: {p['panel']}; border: 2px solid {p['ink']}; }}
QMenu::item:selected {{ background: {p['maroon']}; color: {p['cream']}; }}
QLineEdit {{
    background: {p['lcd']}; color: {p['lcd_ink']}; border: 3px solid {p['ink']};
    border-radius: 0; padding: 0 6px;
}}
QToolTip {{ background: {p['panel']}; color: {p['ink']}; border: 2px solid {p['ink']}; }}
"""
