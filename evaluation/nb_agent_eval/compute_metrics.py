# Copyright (c) 2022–2025 China Merchants Research Institute of Advanced Technology Corporation and its Affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate EM/F1 for *datasets inside a model directory* and export metrics to CSV."""

from __future__ import annotations

import re
import csv
import json
import string
import argparse
from typing import Dict, List, Iterable
from pathlib import Path
from collections import Counter

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str | None) -> List[str]:
    return _normalize(text).split()


def exact_match(answer: str | None, refs: List[str]) -> bool:
    norm = _normalize(answer)
    return any(norm == _normalize(r) for r in refs)


def f1_score(answer: str | None, refs: List[str]) -> float:
    if answer is None or not refs:
        return 0.0
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0

    best = 0.0
    for ref in refs:
        ref_tokens = _tokens(ref)
        if not ref_tokens:
            continue
        common = Counter(ans_tokens) & Counter(ref_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(ans_tokens)
        recall = overlap / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def evaluate_file(path: Path) -> Dict[str, float]:
    """Return metrics for a single ``result.jsonl`` file."""
    total_f1 = total_em = 0.0
    samples = 0

    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"[WARN] {path}: line {idx} invalid JSON — skipped")
                continue

            answers = obj.get("answer")
            if not isinstance(answers, list):
                answers = [answers]
            answers = answers or [""]

            labels = obj.get("label", [])
            if not labels:
                print(f"[WARN] {path}: line {idx} missing labels — skipped")
                continue
            f1 = []
            em = []
            for answer in answers:
                f1.append(f1_score(answer, labels))
                em.append(float(exact_match(answer, labels)))

            total_f1 += sum(f1) / len(f1)
            total_em += sum(em) / len(em)
            samples += 1

    if samples == 0:
        return {"avg_f1": 0.0, "avg_em": 0.0, "total_samples": 0}

    return {
        "avg_f1": total_f1 / samples,
        "avg_em": total_em / samples,
        "total_samples": samples,
    }


def _find_dataset_files(model_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield *(dataset_name, result_path)* inside *model_dir*."""
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        result_file = dataset_dir / "result.jsonl"
        if result_file.is_file():
            yield (dataset_dir.name, result_file)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute EM/F1 for every dataset under a model directory and export CSV summary"
    )
    p.add_argument("model_dir", type=Path, help="Path to a model directory (contains dataset sub-dirs)")
    p.add_argument("--csv", type=Path, default=None, help="Destination CSV for aggregated metrics")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    rows: List[Dict[str, str]] = []

    for dataset_name, jsonl_path in _find_dataset_files(args.model_dir):
        print("=" * 60)
        print(f"Evaluating {dataset_name}: {jsonl_path}")
        metrics = evaluate_file(jsonl_path)

        print(f"Samples      : {metrics['total_samples']}")
        print(f"Average F1   : {metrics['avg_f1']:.4f}")
        print(f"Average EM   : {metrics['avg_em']:.4f}")
        print(f"EM Accuracy  : {metrics['avg_em']:.2%}")

        rows.append(
            {
                "dataset": dataset_name,
                "avg_f1": f"{metrics['avg_f1']:.4f}",
                "avg_em": f"{metrics['avg_em']:.4f}",
                "total_samples": str(metrics["total_samples"]),
            }
        )

    # Write CSV
    print("=" * 60)
    if args.csv:
        print(f"Writing CSV summary to {args.csv}…")
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["dataset", "avg_f1", "avg_em", "total_samples"])
            writer.writeheader()
            writer.writerows(rows)
    print("Done.")


if __name__ == "__main__":
    main()
