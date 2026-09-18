#!/usr/bin/env bash
# verify-eval-baseline スキル用の補助スクリプト
# 評価スイートを実行し、ベースラインが未作成なら新規作成する。
#
# 使い方:
#   bash .claude/skills/verify-eval-baseline/run-eval.sh [all|retrieval|llm] [--repeat N]
#
# 判定（合格/不合格/評価不能）はしない。終了コードと status を集めて表示するだけで、
# 解釈は SKILL.md の手順に従って行う。
#
# ベースラインの扱い:
#   - 未作成    -> --update-baseline を付けて新規作成する
#   - 作成済み  -> 通常実行して比較する（比較不能でも自動更新はしない。条件が変わった
#                  理由の判断は人が行うべきで、自動更新すると退行の検知が静かに失われる）

set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1

TARGET="${1:-all}"
REPEAT=3
if [ "${2:-}" = "--repeat" ]; then
    REPEAT="${3:-3}"
fi

BASELINE_DIR="evals/baseline"
RUNNER="local"
[ "${GITHUB_ACTIONS:-}" = "true" ] && RUNNER="github-actions"

declare -A EXIT_CODES=()
declare -A BASELINE_ACTIONS=()

run_suite() {
    local suite="$1"
    local baseline="${BASELINE_DIR}/${suite}.${RUNNER}.json"
    local args=("$suite")

    if [ "$suite" = "llm" ]; then
        args+=("--repeat" "$REPEAT")
    fi

    if [ -f "$baseline" ]; then
        BASELINE_ACTIONS["$suite"]="既存と比較 (${baseline})"
    else
        BASELINE_ACTIONS["$suite"]="新規作成 (${baseline})"
        args+=("--update-baseline")
        echo "[情報] ベースライン未作成のため --update-baseline を付けて実行します: ${baseline}"
        echo "       （絶対要件違反があった場合、ハーネス側の判断で更新はスキップされます）"
    fi

    echo
    echo "=========================================="
    echo "SUITE: ${suite}  (runner=${RUNNER})"
    echo "=========================================="
    uv run python -m evals.run "${args[@]}"
    EXIT_CODES["$suite"]=$?
}

case "$TARGET" in
    all)
        run_suite retrieval
        run_suite llm
        ;;
    retrieval | llm)
        run_suite "$TARGET"
        ;;
    *)
        echo "使い方: run-eval.sh [all|retrieval|llm] [--repeat N]" >&2
        exit 64
        ;;
esac

echo
echo "=========================================="
echo "まとめ"
echo "=========================================="
overall=0
for suite in "${!EXIT_CODES[@]}"; do
    code="${EXIT_CODES[$suite]}"
    case "$code" in
        0) label="✅ 合格" ;;
        1) label="❌ 不合格（絶対要件違反 または ベースラインからの低下）" ;;
        2) label="⚠️ 評価不能（比較が成立していない。合格ではない）" ;;
        *) label="不明な終了コード" ;;
    esac
    echo "- ${suite}: ${label} (exit=${code}) / ベースライン: ${BASELINE_ACTIONS[$suite]}"
    [ "$code" -ne 0 ] && overall=1
done

echo
echo "詳細レポート: var/eval/<suite>.md / var/eval/<suite>.json"
echo "「落ちた質問」欄と「比較条件」の（不一致）表示を必ず読むこと（SKILL.md 手順3・4）。"

exit "$overall"
