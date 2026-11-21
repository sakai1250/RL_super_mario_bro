#!/bin/zsh
# 依存関係を同梱した軽量 .app バンドルを作る（PyInstaller は使わない）
set -euo pipefail

if [ "$(uname)" != "Darwin" ]; then
  echo "このスクリプトは macOS 専用です。" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/Mario RL Player.app"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "先に 'Mario RL Player.command' を実行して .venv を作成してください。" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "アイコンを生成中…"
QT_QPA_PLATFORM=offscreen "$PY" "$ROOT/scripts/make_icon.py" "$WORK/icon_1024.png" >/dev/null

ICONSET="$WORK/AppIcon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$WORK/icon_1024.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  dbl=$((size * 2))
  sips -z "$dbl" "$dbl" "$WORK/icon_1024.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$WORK/AppIcon.icns"

echo "バンドルを作成中…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$WORK/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Mario RL Player</string>
    <key>CFBundleDisplayName</key><string>Mario RL Player</string>
    <key>CFBundleIdentifier</key><string>io.github.sakai1250.mario-rl-player</string>
    <key>CFBundleVersion</key><string>1.0.0</string>
    <key>CFBundleShortVersionString</key><string>1.0.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>MarioRL</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.games</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/MarioRL" <<'LAUNCHER'
#!/bin/zsh
APP_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG="$HOME/Library/Logs/MarioRLPlayer.log"

if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
  /usr/bin/osascript -e 'display alert "Mario RL Player" message "初回セットアップが必要です。Finder で「Mario RL Player.command」をダブルクリックしてから、もう一度開いてください。" as critical'
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
cd "$APP_DIR"
exec "$APP_DIR/.venv/bin/python" "$APP_DIR/run_app.py" >>"$LOG" 2>&1
LAUNCHER

chmod +x "$APP/Contents/MacOS/MarioRL"

# ローカル実行用のアドホック署名（失敗しても続行）
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true

echo "完成: $APP"
