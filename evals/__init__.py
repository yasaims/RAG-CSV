"""回答品質の評価ハーネス。

`safety_qa` パッケージの配布物には含めない開発ツール。`safety_qa.infra`（合成点）・
`safety_qa.core` の両方に依存する消費者であり、依存の向きは常に `evals -> safety_qa`
（逆向きにしない。import-linter の契約で機械的に検証する）。
"""
