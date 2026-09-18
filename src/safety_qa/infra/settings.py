"""環境変数による設定（`SAFETY_QA_` プレフィックス）。

DB ファイル・Ollama 接続・埋め込みモデル等、環境ごとに変わりうる値をここに集約する。
同時実行数上限など実行環境に依存する値は、運用実績を見て見直す前提の暫定値。
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/safety_qa/infra/settings.py -> infra -> safety_qa -> src -> リポジトリルート
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAFETY_QA_", env_file=".env", extra="ignore")

    csv_path: Path = REPO_ROOT / "data" / "accidents.csv"
    # accidents.duckdb は CSV から再構築可能なキャッシュ、queries.duckdb は監査ログ。
    # 再構築時に監査ログを失わないよう別ファイルに分離する。
    accidents_db_path: Path = REPO_ROOT / "var" / "accidents.duckdb"
    queries_db_path: Path = REPO_ROOT / "var" / "queries.duckdb"

    embedding_model_name: str = "cl-nagoya/ruri-v3-130m"
    embedding_dim: int = 512
    # 既定は遅延ロード（差分埋め込みが 0 件の起動時は torch/モデルに触れない）。
    # 起動時にモデルの誤設定を検知したい運用では True にする。
    preload_embedder: bool = False
    # 既定ではサーバー起動時に ETL（CSV -> DuckDB 取り込み）を実行する。
    # 既存の DB をそのまま使う開発時の再起動を速くしたい場合などに true にする。
    skip_etl: bool = False

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:latest"
    # SPEC.md の必須制約（4 コア）に合わせた既定値。
    ollama_num_thread: int = 4
    ollama_timeout_seconds: float = 60.0
    # Ollama サーバーの既定値（2048〜4096）だと、SYSTEM_PROMPT + top_n 件の検索
    # 結果で文脈が黙って切り詰められうるため明示的に確保する。
    ollama_num_ctx: int = 8192
    # 既定の 5 分アイドルでアンロードされると、間隔をおいた次の質問がモデル
    # 再ロード待ちになる。"-1" を指定すると常駐する（メモリとのトレードオフ）。
    ollama_keep_alive: str = "30m"

    top_n: int = 5
    # ローカル LLM への並行リクエスト数上限。
    max_concurrent_answers: int = 1
    # `GET /v1/queries/{id}/records` の1ページあたりの上限件数。フェーズ1の
    # コーパス規模（数十件）を大きく上回る値にし、実質すべてのケースで
    # 1回のリクエストで全件取得できるようにする。
    max_page_size: int = 100

    # 「今年」「去年」等の相対日付表現を解釈する基準。プロセスのローカル TZ
    # （`date.today()`）に依存すると UTC コンテナで「本日」が最大 1 日ずれる。
    # 工場の現場時間に合わせ既定は Asia/Tokyo。
    business_timezone: str = "Asia/Tokyo"

    # 構造化ログ。stdout への JSON Lines 出力が既定。ローカル開発で人間可読な
    # 出力にしたい場合は console を指定する。
    log_level: str = "INFO"
    log_format: str = "json"
    # 全ログ行に付与する共通フィールド。複数環境のログを1つの CloudWatch
    # ロググループに集約したときの絞り込み用途（将来 AWS へ載せる際の拡張点）。
    service_name: str = "safety-qa"
    environment: str = "local"
    # trace_id の継承元として受信ヘッダ（traceparent 等）を信頼するか。
    # 公開 API で無条件に信じるとログ ID の偽装経路になるため既定 false。
    # 信頼できるプロキシ（ALB 等）配下に置く場合のみ true にする。
    trust_inbound_trace_header: bool = False


def load_settings() -> Settings:
    return Settings()
