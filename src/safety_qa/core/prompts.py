"""LLM に渡すプロンプト文字列、および回答文に付す決定的な文言。

プロトタイプ検証（docs/stack-feasibility-report.md D 節）で、「傾向」を問う
分析系の質問に対しツールを呼ばずに数値・傾向を捏造する失敗モードが
再現性をもって観測された。以下はその対策を反映している:

1. ツール引数スキーマに shift（日勤/夜勤）を含める（core/tools.py 側の対応）
2. 「必ずツールを呼ぶ」「根拠のない数値・傾向の生成を禁止」を明示（本ファイル）
3. 「傾向」「なぜ」等の分析的な質問を search_accident_records に倒す例を明示する
4. temperature=0 を明示的に固定する（infra/llm/ollama_chat.py 側の対応）
5. ツール未呼び出し時は2パス目のLLM生成そのものを行わず、対応できない旨の
   固定文を返す（core/rag_service.py 側の対応。NO_TOOL_CALL_ANSWER を参照）

回答が静かに誤る経路を塞ぐため、以下も持つ:

6. 現在日付をシステムプロンプトへ注入し、相対日付の解釈を安定させる
   （`build_system_prompt`。infra/llm/ollama_chat.py が呼び出しごとに組み立てる）
7. 収録期間外の 0 件を「0件です」と断定させない固定文（`out_of_coverage_answer`）
8. 解釈済み条件・参照した件数・収録期間を、LLM を通さず回答文へ決定的に付す
   フッター（`build_answer_basis_footer`）。母数を2パス目のLLMの文脈には渡さない
   （渡すと自由記述の中で誤転写されうるため）
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from uuid import UUID

from safety_qa.core.filters import AXIS_VALUE_EXAMPLES, RecordFilter

SYSTEM_PROMPT_TEMPLATE = """\
あなたは工場の安全管理担当者向けの Q&A アシスタントです。
事故・ヒヤリハット記録データベースを参照して質問に答えます。

本日は {today} です。「今年」「去年」「来年」「直近半年」「先月」等の相対的な期間表現は、
必ずこの日付を基準に絶対日付（date_from / date_to）へ変換してツールに渡してください。
基準日を明示せずに年・期間を推測しないこと。

**質問文に明示されていない絞り込み条件を、推測やツールの引数例から補ってはいけません。**
期間（date_from / date_to）は、質問文に期間の指定（絶対日付・相対表現のいずれも）が
無い場合は一切渡さないこと。期間を渡さなければ全期間（データベースの収録期間全体）が
対象になります。期間の指定が無いのに実行日を基準に期間を補って渡すと、実際には
該当記録があるのに「収録期間外」と誤判定される原因になります。
場所・設備等の軸も同様に、質問文に明示された語だけを渡してください。
「同じ場所で」「どこで」のように場所を*問う*表現は、場所の*指定*ではありません。

**あなた自身は事故記録の内容を一切知りません。** データベースを検索・集計した結果だけが
唯一の情報源です。想像や一般論で数値・傾向・件数を作り出すことは固く禁止します。

質問には必ずいずれか1つのツールを呼び出してから回答してください。
- 記述内容・傾向・状況（どんな事故か、原因、対策、「傾向」「特徴」「なぜ」を問う質問も含む）
  -> search_accident_records（重大度では絞り込めません。重大度を表す語が質問文に
  あっても query に含めるだけにしてください）
- 件数・集計を問う質問 -> aggregate_accident_records

例:
- 「夜勤帯に起きた旋盤の事故にはどんな傾向がありますか」-> search_accident_records
- 「なぜプレス機の事故が多いのですか」-> search_accident_records
- 「経験3年未満の作業員の事故事例を教えて」-> search_accident_records（experience_max=2）
- 「フォークリフトのヒヤリハットは同じ場所で繰り返し発生していませんか」
  -> search_accident_records（query のみ。具体的な場所名が質問文に無いため
  location は渡さない）
- 「高所作業車での作業中に転落しそうになった事例の傾向を教えて」
  -> search_accident_records（equipment=高所作業車。「高所作業車」は設備であり
  場所ではないため location には渡さない）
- 「重大事故は何件ですか」-> aggregate_accident_records（severity=重大。期間の指定が
  無いため date_from / date_to は渡さない）
- 「溶接機に関する事故は全部で何件ですか」-> aggregate_accident_records（equipment=溶接機。
  期間の指定が無いため date_from / date_to は渡さない）
- 「2025年の重大事故は何件ですか」-> aggregate_accident_records
  （date_from=2025-01-01, date_to=2025-12-31, severity=重大）
- 「今年の重大事故は何件ですか」-> aggregate_accident_records
  （date_from={today_year_start}, date_to={today_year_end}, severity=重大）
- 「来年の重大事故は何件ですか」-> aggregate_accident_records
  （date_from={next_year_start}, date_to={next_year_end}, severity=重大）
- 「夜勤帯の重大事故は何件ですか」-> aggregate_accident_records（shift=夜勤, severity=重大。
  「夜勤帯」は勤務区分なので shift を使い、hour_from / hour_to は渡さない）
- 「深夜帯の事故は何件ですか」-> aggregate_accident_records（hour_from=22, hour_to=5。
  「深夜帯」は時刻表現なので hour_from / hour_to を使い、shift は渡さない。
  shift と hour_from / hour_to を同時に渡さないこと）
- 「中位以上の事故は何件ですか」-> aggregate_accident_records（severity_min=中位）
- 「第2製造棟 機械加工エリアのヒヤリハットは何件ですか」-> aggregate_accident_records
  （severity=ヒヤリハット。location は「第2製造棟 機械加工エリア」のまま渡し、
  棟とエリアに分割しない。期間の指定が無いため date_from / date_to は渡さない）

aggregate_accident_records は件数の集計のみに対応します。平均・合計・割合・比率等の
算出を尋ねられた場合は、ツールを呼び出さずに「その集計はできません（件数のみ算出できます）」
と回答してください。
"""


def build_system_prompt(today: date) -> str:
    """実行時点の日付を注入したシステムプロンプトを組み立てる。

    定数化すると相対日付（「今年」等）の解釈が LLM の推測に委ねられ、
    プロセス常駐中の日付変化にも追随できないため、呼び出しごとに組み立てる。
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        today=today.isoformat(),
        today_year_start=date(today.year, 1, 1).isoformat(),
        today_year_end=date(today.year, 12, 31).isoformat(),
        next_year_start=date(today.year + 1, 1, 1).isoformat(),
        next_year_end=date(today.year + 1, 12, 31).isoformat(),
    )


ANSWER_PROMPT_TEMPLATE = """\
以下は質問に対してデータベースから取得した結果です。この結果に基づいてのみ、
日本語で簡潔に回答してください。結果に含まれない情報を推測で補わないでください。

質問: {question}

取得結果:
{context}
"""

# 対策5: ツール未呼び出し時に質問文を2パス目のLLMに渡して回答文を生成させると、
# 固定文脈に差し替えても「モデルに自由生成させる」経路自体は残り、プロンプトや
# モデルの変更で捏造が再発しうる（プロトタイプで再現性をもって観測された最重大の
# 失敗モード）。2パス目のLLM呼び出しそのものを行わず固定文を返すことで、
# 捏造の発生経路を構造的に塞ぐ（core/rag_service.py 参照）。
NO_TOOL_CALL_ANSWER = (
    "その質問には対応できません。事故記録の記述内容の検索、または"
    "重大度・期間・設備・場所・勤務帯・職種・経験年数・時間帯による件数の集計"
    "（件数のみ。平均・合計・割合は不可）であれば回答できます。"
)


# 複合地名（例:「第2製造棟 機械加工エリア」）を location/equipment に誤分割すると、
# 片方の軸が実データに存在しない値になり AND 条件で静かに 0 件になる。「本当に0件」と
# 「解釈がずれた」を利用者が区別できないため、対策5と同じ考え方で、単体では1件も
# 一致しない軸（空振り軸）がある場合は2パス目のLLM生成を行わず固定文を返す。
def unmatched_filter_answer(record_filter: RecordFilter, unmatched_axes: Sequence[str]) -> str:
    """絞り込み条件のうち単体でも一致しない軸がある場合の固定回答文を組み立てる。"""
    axis_descriptions = "、".join(record_filter.only(axis).describe() for axis in unmatched_axes)
    hints = "。".join(
        f"{axis}の実データの例: " + "、".join(AXIS_VALUE_EXAMPLES[axis])
        for axis in unmatched_axes
        if axis in AXIS_VALUE_EXAMPLES
    )
    # 数字表記を含めない。evals/metrics.py の件数抽出パターン（`\d+\s*件`）が
    # 「0件」を件数言及として拾い、安全側の縮退を捏造・偽ゼロと誤検出するため。
    # 「存在しない値」ではなく「この表記に一致しない」と書く。前者は現場語の
    # 表記ゆれ（例:「1棟」。実データは「第1製造棟」で本文には「1棟」の記述もある）
    # に対して、実在する場所を「存在しない」と誤って断言してしまうため。
    message = (
        f"指定された条件のうち「{axis_descriptions}」はこの表記に一致する記録がないため、"
        "件数を判断できません（該当が無いと確定したわけではありません）。"
    )
    if hints:
        message += f" {hints}。"
    message += "場所・設備・職種等の表現を見直して再度お尋ねください。"
    return message


# 相対日付を絶対日付へ変換できても、データの収録期間を跨いだ期間を指定すると
# 常に0件になる。これを「0件です」と断定すると、利用者は「事故が無かった」のか
# 「データが尽きている」のか区別できない（空振り軸と同じ扱いを日付軸に適用）。
def out_of_coverage_answer(record_filter: RecordFilter, data_period: tuple[date, date]) -> str:
    """指定された期間が収録期間の外にある場合の固定回答文を組み立てる。

    2パス目のLLM生成は行わない。文中に「N件」等の数字表記を含めない
    （`unmatched_filter_answer` と同じ理由。evals/metrics.py の件数抽出パターンが
    拾い、絶対要件2/3を誤って発火させるため）。
    """
    first, last = data_period
    return (
        f"指定された期間（{record_filter.describe()}）はデータベースの収録期間"
        f"（{first.isoformat()}〜{last.isoformat()}）の外にあるため、"
        "件数を判断できません（この期間の記録が登録されていないだけで、"
        "事故が発生しなかったという意味ではありません）。"
        "データの更新状況をご確認のうえ、収録期間内の日付で再度お尋ねください。"
    )


# 解釈済みの条件・母数・収録期間を、利用者が回答文だけで検証できるようにする。
# `RagQaService` がツール実行結果から決定的に組み立て、LLM の生成には含めない
# （母数を2パス目の文脈に渡すと自由記述の中で誤って転写されうる）。
ANSWER_BASIS_HEADING = "【回答の根拠】"


def _describe_ignored_axis(key: str, value: str) -> str:
    """`ignored_axes` の1エントリを人が読める形にする（search限定の軸のみ）。"""
    if key == "severity_min":
        return f"重大度={value}以上"
    return f"重大度={value}"


def build_answer_basis_footer(
    *,
    record_filter: RecordFilter,
    group_by: str | None,
    matched_count: int,
    cited_count: int,
    data_period: tuple[date, date],
    today: date,
    query_id: UUID,
    severity_breakdown: dict[str, int] | None = None,
    ignored_axes: tuple[tuple[str, str], ...] = (),
) -> str:
    """回答文に付す決定的な根拠フッターを組み立てる。

    ツールを1回でも実行した場合に呼ぶ（コーパスが空なら呼び出し前に
    `AnswerUnavailableError` へ倒すため、`data_period` は常に取得済み）。

    - 「解釈した条件」は常に出す（0件時も、利用者が解釈のずれを検証できるように）
    - `ignored_axes` が非空なら、LLM が渡したが実行に使わなかった条件
      （`search_accident_records` の `SEARCH_EXCLUDED_FILTER_KEYS`）を1行足す。
      2パス目のLLMの文脈には渡さない（ADR 009 D3。断り書きを文脈に混ぜると
      自由記述の中で誤転写・欠落しうるため、決定的経路だけで運ぶ）
    - 「参照範囲」は `matched_count` が 0 より大きいときのみ出す（0件の数字表記を
      作らないため。`unmatched_filter_answer`/`out_of_coverage_answer` と同じ理由）
    - `severity_breakdown` を渡すと重大度の内訳を1行足す。severity で絞り込んで
      いない集計（例:「溶接機に関する事故は全部で何件ですか」）は、母数にヒヤリ
      ハットと負傷事故が無告知で混在しうる（`_execute_aggregate` 参照）。search
      でも同様に、引用した記録の重大度が一目で分かるよう内訳を出す
      （`search_accident_records` は重大度で絞り込まないため）。LLM に内訳の
      数値を作らせず、citations と同じ母集団から決定的に組み立てる。
      未参照分がある（打ち切りがある）場合は「引用した記録の」内訳であることを
      明示する（母集団全体の内訳と誤読させないため。打ち切りが無ければ引用＝
      母集団なので通常の見出しにする）
    - 未参照分がある場合、`GET /v1/queries/{id}/records` への導線を1行足す
      （issue #23。API フィールドだけでは回答文しか読まない利用者に届かない
      という ADR 009 の教訓と同じ理由。数字表記を含まないため件数照合には
      引っかからない）
    - 「記録の収録期間」は常に出す。傾向を問う質問（母数はあるが日付軸は無い）でも
      「データが古い」事実が伝わるようにする
    """
    condition = record_filter.describe()
    if group_by is not None:
        condition += f"（内訳={group_by}）"
    lines = [ANSWER_BASIS_HEADING, f"解釈した条件: {condition}"]

    if ignored_axes:
        ignored_text = "、".join(_describe_ignored_axis(key, value) for key, value in ignored_axes)
        lines.append(
            f"絞り込みに使わなかった条件: {ignored_text}"
            "（記述内容の検索では重大度で絞り込みません。重大度を問わず関連する記録を含めています）"
        )

    if matched_count > 0:
        scope = "全記録" if record_filter.is_empty else "条件に合致する記録"
        truncated = max(matched_count - cited_count, 0)
        if truncated > 0:
            lines.append(
                f"参照範囲: {scope} {matched_count} 件のうち関連度上位 {cited_count} 件"
                f"（{truncated} 件は未参照）"
            )
            lines.append(f"未参照分の取得: GET /v1/queries/{query_id}/records")
        else:
            lines.append(f"参照範囲: {scope} {matched_count} 件をすべて確認")

        # 複数の重大度が混在する場合のみ出す（severity で絞り込んでいれば単一
        # 値になり、見出しの件数と重複するだけの冗長な行になるため）。
        if severity_breakdown and len(severity_breakdown) > 1:
            breakdown_text = "、".join(
                f"{severity} {count}件" for severity, count in severity_breakdown.items()
            )
            label = "重大度内訳" if truncated == 0 else "引用した記録の重大度内訳"
            lines.append(f"{label}: {breakdown_text}")

    first, last = data_period
    elapsed_days = max((today - last).days, 0)
    lines.append(
        f"記録の収録期間: {first.isoformat()}〜{last.isoformat()}（最終記録から{elapsed_days}日経過）"
    )

    return "\n".join(lines)


def split_answer_basis(answer: str) -> tuple[str, str | None]:
    """回答文を「LLM生成部分」と「根拠フッター」に分離する。

    評価ハーネス（evals/run.py）・捏造検出（evals/metrics.py）はフッター
    （決定的に組み立てた事実であり LLM の生成物ではない）を含めずに判定すべき
    ため、分離してから使う。
    """
    marker = f"\n\n{ANSWER_BASIS_HEADING}"
    idx = answer.find(marker)
    if idx == -1:
        return answer, None
    return answer[:idx], answer[idx + 2 :]
