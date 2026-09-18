"""fugashi（MeCab + unidic-lite）による日本語トークナイズ。

bm25s は英語圏中心のトークナイズを前提とするため、事前分割したトークン列を
そのまま渡す（bm25s.tokenize の既定処理はバイパスする）。
"""

from __future__ import annotations

import threading
import unicodedata

from fugashi import Tagger

# 助詞・助動詞・記号は BM25 のキーワード一致に寄与しないため除外する。
# 「プレス機」「フォークリフト」等の固有語彙は名詞・複合語として残る。
_EXCLUDED_POS = {"助詞", "助動詞", "記号", "補助記号"}

_tagger: Tagger | None = None
# fugashi（MeCab）の Tagger は内部のラティスを共有しており、複数スレッドから
# 同時に呼び出すのは安全でない（SAFETY_QA_MAX_CONCURRENT_ANSWERS を 1 より
# 大きくした場合に検索がスレッドごとに並行実行されうる）。タガーの生成から
# 解析結果（surface）の取り出しまでをロックで直列化する。
_lock = threading.Lock()


def _get_tagger() -> Tagger:
    # モジュール import 時に MeCab 辞書をロードすると、torch を必要としない
    # 軽量なユニットテストまで巻き込むため遅延初期化する。
    global _tagger
    if _tagger is None:
        _tagger = Tagger()
    return _tagger


def tokenize(text: str) -> list[str]:
    # 半角カナ・全角英数字などの表記ゆれを NFKC で正規化し、英字は小文字化する。
    # 索引時・検索時の双方がこの関数を通るため、正規化は常に一致する
    # （例: 「ﾌｫｰｸﾘﾌﾄ」と「フォークリフト」、「PRESS」と「press」が一致する）。
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    with _lock:
        for word in _get_tagger()(normalized):
            pos = word.feature.pos1
            if pos in _EXCLUDED_POS:
                continue
            surface = word.surface.strip()
            if surface:
                tokens.append(surface)
    return tokens
