---
name: verify-answer-against-data
description: 実 Ollama を起動して Q&A API に質問を投げ、回答文中の件数・内訳・絞り込み結果が実データ（AI/data/accidents.csv）と一致するかを独立に検証するスキル。RagQaService の citations はツール実行結果から決定的に採用される（ADR 006）が、answer 本文は LLM の自由生成であり件数の捏造・取り違えが起きうる（docs/stack-feasibility-report.md、issue #16 の実例）。「動作確認して」「実際に質問して確認して」「回答の件数が正しいか検証して」「手動で確認して」「実LLMで試して」といった依頼、または core/filters.py・core/tools.py・core/rag_service.py・infra/duckdb_store 配下の変更後の受け入れ確認で使う。pytest によるユニットテスト検証だけで完了とせず、実 Ollama を経由した end-to-end の答え合わせが必要な場面が対象。
---

# Verify Answer Against Data

Q&A API（`POST /v1/queries`）に実際に質問し、返ってきた回答文・`citations` が
`AI/data/accidents.csv`（正データ、ADR 003）の内容と矛盾しないかを検証する。

## 基本方針

- `answer` 本文は LLM の自由生成なので、件数や内訳の数値を鵜呑みにしない。
  必ず `citations`（ツール実行結果から決定的に採用される値、ADR 006）と
  独立に計算した ground truth の両方と突き合わせる。
- ground truth は常に `AI/data/accidents.csv` から計算する。
  `var/accidents.duckdb` はサーバー起動中は他プロセスからロックされて
  直接開けないことがある（このスキルの検証中に実際に発生した）うえ、
  CSV こそが正データ（README・ADR 003）なので、素直に CSV を読む方が
  速く確実。
- 使い捨ての追加調査コードが必要になった場合も、`python -c` の複数行
  ワンライナーではなく `.claude/tmp/` に `.py` として書き出して実行する
  （ユーザーのグローバル方針）。このスキルの補助スクリプトは
  再利用前提のためスキルディレクトリ側に置いている。

## 手順

### 1. 前提確認

```bash
ollama list   # hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF が pull 済みか確認
```

未取得なら `ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF` を先に行う（README 参照）。

### 2. サーバー起動

```bash
bash .claude/skills/verify-answer-against-data/dev_server.sh start [PORT]
```

- Ollama サーバーが未起動なら起動し、`/api/version` をポーリングして起動を待つ。
- API サーバー（uvicorn）を指定ポート（省略時 8123）で起動し、起動時 ETL 完了を
  `/docs` へのポーリングで待つ（固定 `sleep` で決め打ちしない。ETL は初回数十秒
  かかることがある）。
- `--reload` は付けない（検証中にファイル変更で再起動されると挙動が読めなくなるため）。

### 3. 質問して回答と citations を取得

```bash
python .claude/skills/verify-answer-against-data/ask_and_check.py "<質問文>" --url http://localhost:8123
```

回答全文、`citations` 件数、citations に含まれる `severity`/`equipment` の集合、
citation 一覧（id・日付・設備・場所・重大度）を表示する。
`--expect-count N` を付けると件数が一致しない場合に非 0 終了する。

### 4. CSV から ground truth を計算

```bash
python .claude/skills/verify-answer-against-data/csv_ground_truth.py \
    --severity-min 中位 --group-by equipment
```

`core.filters.RecordFilter` と同じ 12 軸（severity, severity_min, date-from,
date-to, equipment, location, shift, worker-role, experience-min,
experience-max, hour-from, hour-to）を CLI オプションとして受け付け、
該当件数・id 一覧・（`--group-by` 指定時は）`core.aggregation` と同じ並び順
（時系列軸は key 昇順、それ以外は件数降順→key 昇順）で内訳を表示する。
`shift` は CSV に列がないため `infra/ingest.py` の `_derive_shift`
（20:00〜翌6:00 を夜勤とする）と同じロジックで `time` から導出する。

### 5. 突き合わせパターン

| パターン | 確認内容 |
|---|---|
| 件数一致 | `ask_and_check.py` の citations 件数 == `csv_ground_truth.py` の該当件数 |
| 条件充足 | citations 一覧の属性（severity・equipment 等）が要求条件を満たしている（over-match が無い）。ただし `search_accident_records` は `top_n`（既定5）で打ち切られるため、件数が top_n 未満のときだけ厳密一致を期待できる。top_n 以上ヒットしうる質問は `aggregate_accident_records` に倒れる言い方にするか、上位 N 件が条件を満たしているかだけを確認する |
| 内訳一致 | 回答文中の内訳（例:「プレス機: 8件」）の合計が citations 件数と一致し、各キーの値が `csv_ground_truth.py --group-by` の結果と一致する |

### 6. 後片付け

```bash
bash .claude/skills/verify-answer-against-data/dev_server.sh stop [PORT]
```

止め忘れると、次回起動時のポート衝突や `var/accidents.duckdb` のロック残留の
原因になるため必ず実行する。

## 注意事項

- Windows（Git Bash）環境では `pkill` が使えないことがある。`dev_server.sh`
  はその場合 `netstat -ano` でポートの PID を特定して `taskkill //F //PID`
  する経路と、Ollama 自体は `taskkill //F //IM ollama.exe` で止める経路に
  フォールバックする。
- `scripts/ask.py`（動作確認用 CLI）で人間向けに確認するだけでなく、
  `ask_and_check.py` で JSON を構造的に扱うことで、citations の属性チェックや
  `--expect-count` によるスクリプト内アサーションができる。
- 検証対象が P1（core/filters.py・core/tools.py・core/aggregation.py・
  infra/duckdb_store）の変更であれば、新しく追加した軸ごとに最低 1 問は
  実際に投げて確認する。ユニットテストが通っていても、LLM のツール引数生成
  そのものが不安定な場合がある（docs/stack-feasibility-report.md D 節、issue #16）。
