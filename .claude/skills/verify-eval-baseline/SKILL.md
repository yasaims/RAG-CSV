---
name: verify-eval-baseline
description: ローカルで回答品質の評価スイート（uv run poe eval retrieval / llm）を実行し、環境別ベースライン（evals/baseline/<suite>.<runner>.json）と比較して退行の有無を判定するスキル。ベースラインが未作成なら新規作成する。絶対要件違反ゼロ・ベースライン非退行の両方を確認できて初めて「検証済み」とする。RAG パイプライン（core/prompts.py・core/tools.py・core/filters.py・core/rag_service.py・core/aggregation.py・infra/search/・infra/llm/・evals/ 配下、および AI/data/accidents.csv）を変更した後の受け入れ確認、PR 作成前のセルフチェック、「評価を回して」「退行していないか確認して」「ベースラインと比較して」「eval して」といった依頼で使う。個々の回答内容の答え合わせ（実データとの突き合わせ）が目的の場合は verify-answer-against-data を使う。
---

# Verify Eval Baseline

RAG パイプラインへの変更が、**絶対要件を破っていないか**と**既存の実測水準から退行していないか**を、
評価ハーネス（`evals/`、ADR 007）で機械的に確認する。

## 基本方針

- **「緑」は3値ではなく3状態**。終了コード 0（合格）/ 1（不合格）/ 2（**評価不能**）を区別する。
  評価不能は「品質が悪い」ではなく「判定できなかった」であり、**合格として扱わない**
  （ADR 007「比較不能を明示する」／SPEC.md「障害を無言で握り潰さない」）。
- **赤を緑にするために `--update-baseline` を使わない**。ベースライン更新は「今の水準を
  基準として採用する」という判断であり、退行を消す道具ではない。更新は人が明示的に決める
  （ADR 007「更新手続き」）。
- **閾値を勝手に決めない**（AGENTS.md）。合否は固定閾値ではなく「ベースラインからの低下」で決まる。
- ベースラインは**環境別**（`evals/baseline/<suite>.<runner>.json`）。`local` は開発者個人の参考用で
  `.gitignore` 済み、`github-actions` のみコミットされ CI のゲートに使われる。
  ローカルで緑でも、CI 側のベースラインは別物である点に注意する。

## 手順

### 1. 一括実行（推奨）

```bash
bash .claude/skills/verify-eval-baseline/run-eval.sh all --repeat 3
```

- スイートごとにベースラインの有無を調べ、**無ければ `--update-baseline` を付けて新規作成**する。
- 既にあれば通常実行し、終了コードと `status` を回収してまとめて表示する。
- `all` / `retrieval` / `llm` を指定できる。`--repeat` は llm スイートにのみ効く（既定 3）。

個別に叩く場合:

```bash
uv run poe eval retrieval              # Ollama 不要。検索精度のみ（毎 PR 想定）
uv run poe eval llm --repeat 3         # 要 Ollama。ルーティング＋接地性
```

### 2. 前提（llm スイートのみ）

```bash
ollama list   # hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF が pull 済みか
```

未取得なら `ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF`（README 参照）。
llm スイートは 22 ケース × `--repeat` 回の推論を行うため数分かかる。

`--repeat` は **2 以上**にする。1 回だと `routing_stability` が数学的に必ず 1.00 になり、
ベースライン比較の対象から外れる（`evals/run.py` の `_MIN_REPEAT_FOR_STABILITY`）。

### 3. 結果を判定する

出力は `var/eval/<suite>.json`（機械可読）と `var/eval/<suite>.md`（Job Summary 用）。

| 表示 | 終了コード | 意味 | 取るべき行動 |
|---|---|---|---|
| ✅ 合格 | 0 | 絶対要件違反ゼロ、かつベースラインから低下なし | 完了してよい |
| ❌ 不合格 | 1 | 絶対要件違反あり、またはベースラインから低下 | **原因を直す**。ベースラインを書き換えて回避しない |
| ⚠️ 評価不能 | 2 | ベースライン未設定／比較条件の不一致／依存の取得失敗／ゲート対象0件 | 下記「評価不能の切り分け」へ |

「落ちた質問」欄には期待した記録 id・実際に返った id・実引数が出る。
数値だけでなくこの欄を必ず読む（ADR 007「数値ではなく落ちた質問を主役にする」）。

### 4. 評価不能の切り分け

レポート冒頭の「評価不能の理由」と「比較条件」の `（不一致）` 表示を見る。

- **ベースライン未設定（初回実行）**: `run-eval.sh` が自動で作成する。手動なら
  `uv run poe eval <suite> --update-baseline`。ただし**絶対要件違反があると更新されない**
  （その場合はまず違反を直す）。
- **`dataset_version` の不一致**: `evals/dataset/*.yaml` か `evals/schema.py` の `EvalCase`
  を変更した。意図した変更なら `--update-baseline` で採り直す。
  `EvalCase` にフィールドを足すと**全ケースのハッシュが動く**ため、内容を変えていなくても不一致になる。
- **`corpus_version` の不一致**: `AI/data/accidents.csv` が変わった。意図した変更なら採り直す。
- **`model` / `top_n` / `repeat` / `evaluation_date` の不一致**: 実行条件が違う。
  同じ条件で測り直すか、条件変更が意図したものなら採り直す。
- **依存の取得失敗**（Ollama 不達・埋め込みモデル取得失敗）: 品質の問題ではないので直してから再実行する。

### 5. 報告する

```
## 評価結果

### retrieval（Ollama 不要）
- status / 終了コード
- 集計指標の現在値とベースライン値（低下の有無）

### llm（--repeat N）
- status / 終了コード
- 絶対要件違反の件数と内容（落ちた質問の id・原因）
- 集計指標の現在値とベースライン値

### ベースラインの扱い
- 新規作成した／既存と比較した／更新していない のいずれかを明記し、更新した場合はその理由

### 結論
- 検証済みと判断した根拠、または未解決の問題
```

## 注意事項

- **評価不能を「問題なし」と報告しない。** ADR 007 がこの状態をわざわざ終了コード 2 として
  区別したのは、沈黙のスキップを防ぐため。「比較できなかった」と明記する。
- ローカルのベースライン（`*.local.json`）は `.gitignore` 済みでコミットされない。
  CI のゲートに効くのは `*.github-actions.json` のみで、これは CI 上で
  `--update-baseline` を実行したときにしか更新できない。ローカルから書き換えない。
- `SAFETY_QA_SKIP_ETL` を有効にしていると CSV の変更が DB に反映されないまま評価してしまう。
  `AI/data/accidents.csv` を変更した直後は ETL を通す（`uv run poe etl`）。
- llm スイートは `temperature=0` でも実行ごとに結果が揺れることがある
  （プロンプトの長さ・例示の並び順が変わると、無関係なケースの出力まで動く実測がある）。
  1 回の結果で断定せず、赤が出たら同じ条件でもう一度回して再現するか確かめる。
- プロンプト（`core/prompts.py`）を変更したときは、**変更した箇所と無関係なカテゴリ**
  （aggregate / temporal / unsupported）まで必ず確認する。few-shot の追加・並び替えが
  他カテゴリのルーティングを壊した実例がある（ADR 011 の経緯）。
- ユニットテスト（`uv run poe test`）はこのスキルの代わりにはならない。逆も同じで、
  評価が緑でも `pre-commit run --all-files` は別途通すこと。
