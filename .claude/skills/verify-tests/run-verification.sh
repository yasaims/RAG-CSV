#!/usr/bin/env bash
# verify-tests スキル用の補助スクリプト
# テスト実行 → diff概観 → 怪しいパターンのgrep をまとめて行い、
# Claudeが読みやすい形でまとめて出力する。
#
# 使い方:
#   scripts/run_verification.sh [比較対象のgit ref (デフォルト: HEAD~1)]
#
# 出力はすべて標準出力に流れるので、そのままClaudeに読ませて判断材料にする。
# このスクリプト自体はPASS/FAILの判定はしない(判定はSKILL.mdの手順に従って人間/Claudeが行う)。

set -uo pipefail

BASE_REF="${1:-HEAD~1}"

echo "=========================================="
echo "STEP 1: pytest実行結果"
echo "=========================================="
if command -v pytest >/dev/null 2>&1; then
    pytest -q --cov 2>&1 || pytest -q 2>&1
else
    echo "[警告] pytest が見つかりません。仮想環境の有効化やインストールを確認してください。"
fi

echo
echo "=========================================="
echo "STEP 2: 差分の概観 (対象: ${BASE_REF})"
echo "=========================================="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --stat "${BASE_REF}" 2>&1
else
    echo "[警告] gitリポジトリではないようです。diffの確認は手動で行ってください。"
fi

echo
echo "=========================================="
echo "STEP 3: 弱体化の疑いがあるパターンのgrep"
echo "=========================================="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff "${BASE_REF}" | grep -nE "pytest\.(mark\.)?(skip|xfail)|@skip|assert True|except.*:\s*pass|# noqa|TODO|FIXME|collect_ignore" \
        && echo "↑ 該当箇所あり。references/red-flags.md を参照して精査してください。" \
        || echo "該当パターンなし(ただしこれは機械的なgrepであり、意味的な弱体化までは検出できない点に注意)。"
else
    echo "[スキップ] gitリポジトリではないため実行しません。"
fi

echo
echo "=========================================="
echo "完了。上記の出力を根拠にSKILL.mdの報告フォーマットでPASS/FAILを判定してください。"
echo "=========================================="