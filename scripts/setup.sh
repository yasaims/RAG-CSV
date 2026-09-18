#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "== uvをインストールします =="
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "== uv sync =="
uv sync

echo "== pre-commit hooks 有効化 =="
uv run poe hooks

if [ ! -f .env ]; then
  echo "== .env 作成 =="
  cp .env.example .env
else
  echo "== .env は既に存在するためスキップ =="
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "== ollamaをインストールします =="
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "== ollama インストール済み =="
fi

if command -v ollama >/dev/null 2>&1; then
  echo "== Ollama モデル取得 =="
  # systemdがある環境ではinstall.shで自動起動するが、念のため
  ollama serve &>/dev/null &
  ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
else
  echo "警告: ollama が見つかりません。https://ollama.com/ からインストールしてください。" >&2
fi


# 埋め込みモデル（`cl-nagoya/ruri-v3-130m`）は初回起動時に Hugging Face Hub から自動取得される
if [ ! -f var/accidents.duckdb ]; then
  echo "== データ取り込み =="
  uv run poe etl
else
  echo "== DBが存在するため、自動取り込み中止 =="
  echo "手動で更新する場合は'uv run poe etl'を実行"
fi

echo "セットアップ完了。"
echo "'ollama serve' を起動した上で 'uv run poe serve' でサーバーを起動できます。"
echo ".envを編集して設定を変更してください。"
