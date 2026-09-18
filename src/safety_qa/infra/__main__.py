"""ETL 単体実行 CLI。

`uv run python -m safety_qa.infra` でサーバーを起動せずに ETL のみ実行できる。
API サーバー起動時にも同じ `build_container` が同じ ETL を実行するため、
ここでは 2 回実行して冪等性・差分埋め込みを確認する用途を想定する。
"""

from __future__ import annotations

from safety_qa.infra.container import build_container


def main() -> None:
    container = build_container()
    result = container.ingest_result
    container.close()
    print(f"取り込み件数: {result.row_count} 件")
    print(
        f"埋め込み実行: {result.embedded_count} 件 / 既存ベクトル再利用: {result.reused_count} 件"
    )
    print(f"CSV から削除された行の反映: {result.deleted_count} 件")


if __name__ == "__main__":
    main()
