"""verify-answer-against-data スキル用の補助スクリプト。

`AI/data/accidents.csv`（正データ）を直接読み、
`core.filters.RecordFilter` と同じ意味の絞り込み・
`core.aggregation` と同じ意味の内訳集計を、サーバーやDBに一切依存せず
独立に計算する。API 経由の回答（citations）と突き合わせる ground truth
として使う。

`var/accidents.duckdb` はサーバー起動中に他プロセスからロックされて
直接開けないことがあるため、常に読み取り可能な CSV を正データとして使う
（README・infra/ingest.py の方針どおり）。

`shift` は CSV に列が無く ETL 時（infra/ingest.py の `_derive_shift`）に
`time` から導出されるため、ここでも同じロジック（20:00〜翌6:00 を夜勤と
する）で導出する。

使い方:
    python csv_ground_truth.py --severity-min 中位
    python csv_ground_truth.py --equipment プレス機 --group-by month
    python csv_ground_truth.py --experience-max 2 --hour-from 22 --hour-to 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TypedDict

CSV_PATH = Path("AI/data/accidents.csv")
VALID_SEVERITIES = ("ヒヤリハット", "軽微", "中位", "重大")
CATEGORICAL_GROUP_BY = ("equipment", "location", "severity", "shift", "worker_role")
TIME_SERIES_GROUP_BY = ("month", "year")
VALID_GROUP_BY = (*CATEGORICAL_GROUP_BY, *TIME_SERIES_GROUP_BY)

# infra/ingest.py の NIGHT_SHIFT_START/END と同じ値（分単位）
_NIGHT_SHIFT_START_MINUTES = 20 * 60
_NIGHT_SHIFT_END_MINUTES = 6 * 60


class Row(TypedDict):
    id: str
    date: str
    time: str
    location: str
    equipment: str
    severity: str
    worker_id: str
    worker_role: str
    worker_experience_years: str
    description: str
    cause: str
    countermeasure: str


def _read_rows(csv_path: Path) -> list[Row]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))  # type: ignore[arg-type]


def _hour(row: Row) -> int:
    return int(row["time"].split(":")[0])


def _minutes(row: Row) -> int:
    h, m = row["time"].split(":")[:2]
    return int(h) * 60 + int(m)


def _derive_shift(row: Row) -> str:
    """infra/ingest.py の `_derive_shift` と同じロジック（20:00〜翌6:00 = 夜勤）。"""
    minutes = _minutes(row)
    if minutes >= _NIGHT_SHIFT_START_MINUTES or minutes < _NIGHT_SHIFT_END_MINUTES:
        return "夜勤"
    return "日勤"


def _matches(row: Row, args: argparse.Namespace) -> bool:
    if args.severity is not None and row["severity"] != args.severity:
        return False
    if args.severity_min is not None:
        allowed = VALID_SEVERITIES[VALID_SEVERITIES.index(args.severity_min) :]
        if row["severity"] not in allowed:
            return False
    if args.date_from is not None and row["date"] < args.date_from:
        return False
    if args.date_to is not None and row["date"] > args.date_to:
        return False
    if args.equipment is not None and args.equipment not in row["equipment"]:
        return False
    if args.location is not None and args.location not in row["location"]:
        return False
    if args.shift is not None and _derive_shift(row) != args.shift:
        return False
    if args.worker_role is not None and args.worker_role not in row["worker_role"]:
        return False
    experience = int(row["worker_experience_years"])
    if args.experience_min is not None and experience < args.experience_min:
        return False
    if args.experience_max is not None and experience > args.experience_max:
        return False
    if args.hour_from is not None and args.hour_to is not None:
        hour = _hour(row)
        if args.hour_from <= args.hour_to:
            if not (args.hour_from <= hour <= args.hour_to):
                return False
        elif not (hour >= args.hour_from or hour <= args.hour_to):
            return False
    elif args.hour_from is not None and _hour(row) < args.hour_from:
        return False
    elif args.hour_to is not None and _hour(row) > args.hour_to:
        return False
    return True


def _group_key(row: Row, group_by: str) -> str:
    if group_by == "month":
        return row["date"][:7]
    if group_by == "year":
        return row["date"][:4]
    if group_by == "shift":
        return _derive_shift(row)
    return str(row[group_by])  # type: ignore[literal-required]


def _print_groups(matched: list[Row], group_by: str) -> None:
    buckets: dict[str, list[Row]] = {}
    for row in matched:
        buckets.setdefault(_group_key(row, group_by), []).append(row)
    items = sorted(buckets.items())
    if group_by not in TIME_SERIES_GROUP_BY:
        items.sort(key=lambda item: -len(item[1]))
    print(f"\n{group_by}別の内訳（件数の多い順、time系は時系列順）:")
    for key, rows in items:
        print(f"  - {key}: {len(rows)} 件")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--severity", choices=VALID_SEVERITIES, default=None)
    parser.add_argument("--severity-min", choices=VALID_SEVERITIES, default=None)
    parser.add_argument("--date-from", default=None, help="YYYY-MM-DD")
    parser.add_argument("--date-to", default=None, help="YYYY-MM-DD")
    parser.add_argument("--equipment", default=None, help="部分一致")
    parser.add_argument("--location", default=None, help="部分一致")
    parser.add_argument("--shift", choices=("日勤", "夜勤"), default=None)
    parser.add_argument("--worker-role", default=None, help="部分一致")
    parser.add_argument("--experience-min", type=int, default=None)
    parser.add_argument("--experience-max", type=int, default=None)
    parser.add_argument("--hour-from", type=int, default=None, help="0-23")
    parser.add_argument("--hour-to", type=int, default=None, help="0-23")
    parser.add_argument("--group-by", choices=VALID_GROUP_BY, default=None)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    args = parser.parse_args()

    if not args.csv_path.exists():
        parser.error(f"{args.csv_path} が見つかりません。リポジトリルートから実行してください。")

    rows = _read_rows(args.csv_path)
    matched = [row for row in rows if _matches(row, args)]

    print(f"該当件数: {len(matched)} / 全体 {len(rows)}")
    print("該当 id:", [row["id"] for row in matched])

    if args.group_by is not None:
        _print_groups(matched, args.group_by)


if __name__ == "__main__":
    main()
