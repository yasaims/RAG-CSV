"""OpenAPI 3.1 スキーマを `openapi/openapi.yaml` に生成する。

pre-commit（コード変更時）と CI（ドリフト検出）から呼ばれる。
改行コードは `.gitattributes` の eol=lf 方針に合わせ LF 固定。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from safety_qa.api.app import app

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "openapi" / "openapi.yaml"


def main() -> None:
    schema = app.openapi()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(schema, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
