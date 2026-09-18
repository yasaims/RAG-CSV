"""verify-answer-against-data スキル用の補助スクリプト。

Q&A API に質問を投げ、回答文と citations の要約を表示する。
answer は LLM の自由生成であり件数を捏造しうるため、citations
（ツール実行結果から決定的に採用される値）を判断材料として
別途表示する。ground truth との突き合わせは csv_ground_truth.py で行う。

使い方:
    python ask_and_check.py "中位以上の事故は何件ですか" --url http://localhost:8123
    python ask_and_check.py "中位以上の事故は何件ですか" --expect-count 13
"""

from __future__ import annotations

import argparse
import sys

import httpx

DEFAULT_URL = "http://localhost:8123"
TIMEOUT_SECONDS = 120.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="質問文")
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"API サーバーの URL（既定: {DEFAULT_URL}）"
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        default=None,
        help="citations 件数の期待値。不一致の場合は非 0 で終了する。",
    )
    args = parser.parse_args()

    try:
        res = httpx.post(
            f"{args.url}/v1/queries", json={"question": args.question}, timeout=TIMEOUT_SECONDS
        )
    except httpx.ConnectError:
        print(f"サーバーに接続できません: {args.url}", file=sys.stderr)
        print("dev_server.sh start でサーバーを起動してください。", file=sys.stderr)
        sys.exit(1)

    if res.status_code >= 400:
        print(f"[{res.status_code}] {res.text}", file=sys.stderr)
        sys.exit(1)

    body = res.json()
    citations: list[dict[str, str]] = body["citations"]

    print("回答:")
    print(body["answer"])
    print()
    print(f"citations件数: {len(citations)}")
    if citations:
        severities = sorted({c["severity"] for c in citations})
        equipments = sorted({c["equipment"] for c in citations})
        print(f"  含まれる severity: {severities}")
        print(f"  含まれる equipment: {equipments}")
        print("  citation 一覧:")
        for c in citations:
            print(
                f"    - {c['id']} {c['date']} {c['equipment']} {c['location']} 重大度:{c['severity']}"
            )

    if args.expect_count is not None:
        if len(citations) != args.expect_count:
            print(
                f"[FAIL] citations件数が期待値と不一致: 期待={args.expect_count} 実際={len(citations)}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[OK] citations件数が期待値と一致: {args.expect_count}")


if __name__ == "__main__":
    main()
