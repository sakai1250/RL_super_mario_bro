"""プロジェクト内のよく使うパス。"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
SCREENSHOT_DIR = PROJECT_DIR / "screenshots"
FONT_DIR = PROJECT_DIR / "assets" / "fonts"
