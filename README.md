# 工場事故Q&Aシステム

CSVの労働災害記録を、自然言語で検索・集計して応答するシステム\

## サーバー起動

```bash
uv run poe serve # Ctrl + Cで停止
```

## 動作確認スクリプト

```bash
# 対話モード（空行または Ctrl+D で終了）
uv run poe ask

# 一回だけ実行
uv run poe ask "プレス機で指を挟んだ事例を教えて"
uv run poe ask --all "2025年の事故を一覧で教えて"

# 過去の問い合わせ一覧（監査ログ）を表示
uv run poe ask --history
```

## テスト

```bash
uv run poe test                # ユニットテスト（Ollama 不要、CI もこちら）
uv run poe test-integration    # 実 Ollama + 実埋め込みモデルを使う end-to-end 疎通テスト
```

`integration` マーカーの除外は `pyproject.toml` の `addopts` に設定済みで、
`uv run pytest` は既定でユニットテストのみを実行する（CI も同じコマンド）。
`tests/integration/` は Ollama の起動（`ollama serve` + モデル pull 済み）を前提とする。

## 回答品質の評価

pytest とは別の独立 CLI。データセット全体の指標集計・ベースラインとの比較を担う。

```bash
uv run poe eval retrieval                            # 検索精度（Ollama 不要。毎 PR）
uv run poe eval llm --repeat 3                       # 回答品質（要 Ollama。対象ファイル変更 PR / 手動実行）
uv run poe eval retrieval --update-baseline          # 実測値を暫定ベースラインとして保存
```

- 結果は `var/eval/<suite>.json`（機械可読）と `var/eval/<suite>.md`（人が読む
  Job Summary 用。落ちた質問の一覧を優先して表示する）に出力する

## 設定

設定は環境変数（または.envファイル）で与える。\
一覧は[設定リファレンス](docs/configuration.md)を参照。

## ETL

csvを正データとし、`var/accidents.duckdb`（CSV から再構築可能な
キャッシュ）へ取り込む \
`SAFETY_QA_SKIP_ETL=false`:サーバー起動時に自動実行

```bash
uv run poe etl
```
