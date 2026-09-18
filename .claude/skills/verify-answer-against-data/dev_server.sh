#!/usr/bin/env bash
# verify-answer-against-data スキル用の補助スクリプト。
# Ollama + API サーバー（uvicorn）を検証用に起動/停止する。
# 固定 sleep で起動完了を決め打ちせず、ヘルスチェックをポーリングする。
#
# 使い方:
#   dev_server.sh start [PORT]   # 既定 PORT=8123
#   dev_server.sh stop  [PORT]

set -uo pipefail

ACTION="${1:-}"
PORT="${2:-8123}"
OLLAMA_LOG="${TMPDIR:-/tmp}/verify_ollama_serve.log"
UVICORN_LOG="${TMPDIR:-/tmp}/verify_uvicorn_${PORT}.log"

usage() {
    echo "使い方: dev_server.sh {start|stop} [PORT]" >&2
    exit 1
}

wait_for_http_200() {
    local url="$1"
    local timeout_seconds="$2"
    local waited=0
    while [ "$waited" -lt "$timeout_seconds" ]; do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)" = "200" ]; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

start() {
    if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "Ollama サーバーを起動します..."
        (ollama serve >"$OLLAMA_LOG" 2>&1 &)
        if ! wait_for_http_200 "http://localhost:11434/api/version" 30; then
            # /api/version は 200 を返さない実装もあるため、接続可否だけ再確認する
            if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
                echo "[エラー] Ollama サーバーの起動確認がタイムアウトしました。ログ: $OLLAMA_LOG" >&2
                exit 1
            fi
        fi
    else
        echo "Ollama サーバーは起動済みです。"
    fi

    echo "API サーバーを起動します（ポート ${PORT}）..."
    (uv run uvicorn safety_qa.api.app:app --port "$PORT" >"$UVICORN_LOG" 2>&1 &)

    echo "起動時 ETL の完了を待機中（初回はモデル取得を含め数十秒かかることがあります）..."
    if wait_for_http_200 "http://localhost:${PORT}/docs" 120; then
        echo "起動完了: http://localhost:${PORT}"
    else
        echo "[エラー] ポート ${PORT} での起動確認がタイムアウトしました。ログ: $UVICORN_LOG" >&2
        exit 1
    fi
}

stop() {
    if command -v pkill >/dev/null 2>&1; then
        pkill -f "uvicorn safety_qa.api.app:app.*--port ${PORT}" 2>/dev/null || true
        pkill -f "ollama serve" 2>/dev/null || true
    else
        # Windows Git Bash には pkill が無いため、ポートから PID を特定して taskkill する
        local pid
        pid=$(netstat -ano 2>/dev/null | grep ":${PORT} " | grep LISTENING | awk '{print $NF}' | head -1)
        if [ -n "${pid:-}" ]; then
            taskkill //F //PID "$pid" 2>/dev/null || true
            echo "API サーバー（PID ${pid}）を停止しました。"
        else
            echo "ポート ${PORT} で待ち受けているプロセスが見つかりませんでした。"
        fi
        taskkill //F //IM ollama.exe 2>/dev/null || true
    fi
    echo "停止処理が完了しました。"
}

case "$ACTION" in
    start) start ;;
    stop) stop ;;
    *) usage ;;
esac
