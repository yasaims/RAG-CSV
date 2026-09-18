"""動作確認用 CLI。

質問を入力すると `POST /v1/queries` を叩き、回答と応答時間を表示する。
サーバーが起動している前提（`uv run uvicorn safety_qa.api.app:app`）。

対話モード:
    uv run python scripts/ask.py

一回だけ実行:
    uv run python scripts/ask.py "プレス機で指を挟んだ事例を教えて"
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 120.0
# `GET /v1/queries/{id}/records` を1回のリクエストで叩く際のページサイズ。
# サーバー側の SAFETY_QA_MAX_PAGE_SIZE で実効上限にクランプされるため、
# 母集団がそれを超える場合は複数ページに分けて取得する。
_RECORDS_PAGE_SIZE = 100


def _print_citation(c: dict) -> None:
    # PR-4（issue #23 D4）で追加される excerpt_field。未導入・不明時は None。
    field_label = f"（{c['excerpt_field']}に該当）" if c.get("excerpt_field") else ""
    print(f"  - {c['id']} ({c['date']} {c['equipment']} {c['location']} 重大度:{c['severity']})")
    print(f"      {c['excerpt']}{field_label}")


def _fetch_records_page(base_url: str, query_id: str, *, offset: int) -> dict | None:
    try:
        res = httpx.get(
            f"{base_url}/v1/queries/{query_id}/records",
            params={"limit": _RECORDS_PAGE_SIZE, "offset": offset},
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.ConnectError:
        print(f"サーバーに接続できません: {base_url}", file=sys.stderr)
        return None
    if res.status_code >= 400:
        print(f"[{res.status_code}] 未参照分の取得に失敗しました: {res.text}", file=sys.stderr)
        return None
    return res.json()


def _fetch_all_records(base_url: str, query_id: str) -> list[dict] | None:
    """`GET /v1/queries/{id}/records` を必要なだけページングして全件集める。"""
    items: list[dict] = []
    offset = 0
    while True:
        page = _fetch_records_page(base_url, query_id, offset=offset)
        if page is None:
            return None
        items.extend(page["items"])
        offset += len(page["items"])
        if not page["items"] or offset >= page["total"]:
            return items


def _print_records(items: list[dict]) -> None:
    print(f"\n未参照分を含む全 {len(items)} 件:")
    for item in items:
        print(
            f"  - {item['id']} ({item['date']} {item['equipment']} {item['location']} "
            f"重大度:{item['severity']})"
        )


def _offer_remaining_records(base_url: str, body: dict, *, show_all: bool) -> None:
    basis = body.get("basis")
    if not basis or basis["truncated_count"] <= 0:
        return

    if not show_all:
        try:
            reply = (
                input(
                    f"\n未参照分が {basis['truncated_count']} 件あります。"
                    "一覧を表示しますか？ [y/N] "
                )
                .strip()
                .lower()
            )
        except EOFError:
            print()
            return
        if reply != "y":
            return

    items = _fetch_all_records(base_url, body["id"])
    if items is not None:
        _print_records(items)


def ask(base_url: str, question: str, *, show_all: bool = False) -> None:
    start = time.perf_counter()
    try:
        res = httpx.post(
            f"{base_url}/v1/queries", json={"question": question}, timeout=TIMEOUT_SECONDS
        )
    except httpx.ConnectError:
        print(f"サーバーに接続できません: {base_url}", file=sys.stderr)
        print(
            "`uv run uvicorn safety_qa.api.app:app` でサーバーを起動してください。", file=sys.stderr
        )
        sys.exit(1)
    elapsed = time.perf_counter() - start

    if res.status_code >= 400:
        print(f"[{res.status_code}] ({elapsed:.1f}秒)")
        print(res.text)
        return

    body = res.json()
    print(f"\n回答（{elapsed:.1f}秒）:")
    print(body["answer"])
    if body["citations"]:
        print("\n参照した事故記録:")
        for c in body["citations"]:
            _print_citation(c)
    else:
        print("\n参照した事故記録: なし")

    _offer_remaining_records(base_url, body, show_all=show_all)


def show_history(base_url: str, *, limit: int, before_id: str | None) -> None:
    params: dict[str, str | int] = {"limit": limit}
    if before_id is not None:
        params["before_id"] = before_id
    try:
        res = httpx.get(f"{base_url}/v1/queries", params=params, timeout=TIMEOUT_SECONDS)
    except httpx.ConnectError:
        print(f"サーバーに接続できません: {base_url}", file=sys.stderr)
        sys.exit(1)

    if res.status_code >= 400:
        print(f"[{res.status_code}]")
        print(res.text)
        return

    body = res.json()
    if not body["items"]:
        print("問い合わせの記録はありません。")
        return

    print(f"\n過去の問い合わせ（新しい順、{len(body['items'])}件）:")
    for item in body["items"]:
        status_label = "失敗" if item["status"] == "failed" else "成功"
        print(f"  - [{item['id']}] {item['created_at']} ({status_label}) {item['question']}")
    if body["next_before_id"] is not None:
        print(f"\nさらに遡る: --history --before-id {body['next_before_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="動作確認用 CLI")
    parser.add_argument("question", nargs="?", help="質問文（省略時は対話モード）")
    parser.add_argument(
        "--url", default=DEFAULT_BASE_URL, help=f"API サーバーの URL（既定: {DEFAULT_BASE_URL}）"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="citations に含まれなかった残りの記録があれば、確認なしで一覧表示する",
    )
    parser.add_argument(
        "--history", action="store_true", help="質問文の代わりに過去の問い合わせ一覧を表示する"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="--history 使用時の件数上限（既定: 20）"
    )
    parser.add_argument(
        "--before-id", help="--history 使用時、この id より古い問い合わせから表示する"
    )
    args = parser.parse_args()

    if args.history:
        show_history(args.url, limit=args.limit, before_id=args.before_id)
        return

    if args.question is not None:
        ask(args.url, args.question, show_all=args.all)
        return

    print(f"対話モード（{args.url}）。空行または Ctrl+D で終了。")
    while True:
        try:
            question = input("\n質問> ").strip()
        except EOFError:
            print()
            break
        if not question:
            break
        ask(args.url, question, show_all=args.all)


if __name__ == "__main__":
    main()
