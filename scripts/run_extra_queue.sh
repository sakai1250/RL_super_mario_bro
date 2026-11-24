#!/bin/bash
# 追加学習キュー: 担当ステージを順に学習し、全てクリアするまで周回する。
# 1ステージあたり1周で最大3区切り（360万ステップ）まで進め、
# クリアしないステージは次の周回で続きを学習する。
# 使い方: CUDA_VISIBLE_DEVICES=0 bash scripts/run_extra_queue.sh 0 1-3 4-3 ...
cd "$(dirname "$0")/.." || exit 1
GROUP=$1; shift
STAGES=("$@")
while true; do
  remaining=0
  for s in "${STAGES[@]}"; do
    if grep -q '"cleared": true' "results/full_run/$s/done.json" 2>/dev/null; then
      continue
    fi
    remaining=1
    echo "[queue$GROUP] training $s $(date '+%F %T')"
    .venv/bin/python scripts/train_extra_stages.py "$s" "$GROUP" 3
  done
  if [ "$remaining" = 0 ]; then
    echo "[queue$GROUP] all stages cleared $(date '+%F %T')"
    exit 0
  fi
  sleep 10
done
