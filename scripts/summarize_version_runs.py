#!/usr/bin/env python3
"""Summarize repeated qwen_decode metrics without editing immutable artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FIELDS = (
    "sparse_tokens_per_s",
    "sparse_mean_decode_ms",
    "sparse_p50_decode_ms",
    "sparse_p95_decode_ms",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.metrics]
    print(f"runs: {len(reports)}")
    for field in FIELDS:
        values = [float(report[field]) for report in reports]
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{field}: {statistics.mean(values):.6f} ± {deviation:.6f}")
    qualities = {
        (report["exact_token_prefix"], round(report["mean_logits_cosine"], 9))
        for report in reports
    }
    print(f"quality tuples (exact prefix, cosine): {sorted(qualities)}")


if __name__ == "__main__":
    main()
