#!/bin/zsh
# ダブルクリックで Mario RL Player を起動するランチャー
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "初回セットアップを開始します…"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv が見つかりません。先に 'brew install uv' を実行してください。"
    read -r "?Enter キーで閉じます…"
    exit 1
  fi
  uv venv --python 3.13 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi

if ! .venv/bin/python -c "import PySide6, torch, gym_super_mario_bros" >/dev/null 2>&1; then
  echo "依存関係をインストールしています…"
  uv pip install --python .venv/bin/python -r requirements.txt
fi

exec .venv/bin/python run_app.py
