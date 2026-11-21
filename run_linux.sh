#!/usr/bin/env bash
# Mario RL Player を Linux で起動するランチャー
set -euo pipefail
cd "$(dirname "$0")"

# gym-super-mario-bros 9.x は Python 3.13 以上が必要。
find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
  elif [ -x "$HOME/.local/bin/uv" ]; then
    echo "$HOME/.local/bin/uv"
  fi
}

if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)'; then
  echo "既存の .venv は Python 3.13 未満です。作り直します…"
  rm -rf .venv
fi

if [ ! -x .venv/bin/python ]; then
  echo "初回セットアップを開始します…"
  UV="$(find_uv || true)"
  if [ -n "$UV" ]; then
    "$UV" venv --python 3.13 .venv
    "$UV" pip install --python .venv/bin/python -r requirements.txt
  elif command -v python3.13 >/dev/null 2>&1; then
    python3.13 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
  else
    echo "Python 3.13 か uv が必要です。"
    echo "  uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  または python3.13 をインストールしてください。"
    exit 1
  fi
fi

if ! .venv/bin/python -c "import PySide6, torch, gym_super_mario_bros" >/dev/null 2>&1; then
  echo "依存関係をインストールしています…"
  UV="$(find_uv || true)"
  if [ -n "$UV" ]; then
    "$UV" pip install --python .venv/bin/python -r requirements.txt
  else
    .venv/bin/python -m pip install -r requirements.txt
  fi
fi

exec .venv/bin/python run_app.py "$@"
