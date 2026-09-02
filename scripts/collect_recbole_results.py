#!/usr/bin/env python3
import argparse
import ast
import csv
import re
from pathlib import Path


RUN_PATTERN = re.compile(
    r"(?P<dataset>.+?)_(?P<model>[^_]+)_L(?P<max_len>\d+)_seed(?P<seed>\d+)"
)


def parse_result_dict(raw):
    raw = raw.strip()
    if raw.startswith("OrderedDict(") and raw.endswith(")"):
        pairs = ast.literal_eval(raw[len("OrderedDict(") : -1])
        return dict(pairs)
    return ast.literal_eval(raw)


def find_last_result(log_text, marker):
    result = None
    for line in log_text.splitlines():
        if marker in line:
            _, raw = line.split(marker, 1)
            raw = raw.strip()
            if raw.startswith(":"):
                raw = raw[1:].strip()
            try:
                result = parse_result_dict(raw)
            except (SyntaxError, ValueError):
                continue
    return result or {}


def parse_run_name(path):
    match = RUN_PATTERN.search(path.stem)
    if not match:
        return {
            "dataset": "",
            "model": "",
            "max_len": "",
            "seed": "",
        }
    return match.groupdict()


def collect(log_dir):
    rows = []
    for path in sorted(Path(log_dir).glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        row = parse_run_name(path)
        row["log_file"] = str(path)

        valid = find_last_result(text, "best valid result")
        test = find_last_result(text, "test result")

        for key, value in valid.items():
            row[f"valid_{key.lower()}"] = value
        for key, value in test.items():
            row[f"test_{key.lower()}"] = value
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = collect(args.log_dir)
    fieldnames = sorted({key for row in rows for key in row})
    priority = ["dataset", "model", "max_len", "seed", "log_file"]
    fieldnames = priority + [key for key in fieldnames if key not in priority]

    output_path = Path(args.output) if args.output else None
    output = output_path.open("w", newline="", encoding="utf-8") if output_path else None
    try:
        target = output if output is not None else __import__("sys").stdout
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
