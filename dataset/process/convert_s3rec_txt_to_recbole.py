#!/usr/bin/env python3
"""Convert S3-Rec sequence txt files into RecBole atomic format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def read_sequences(path: Path) -> Dict[str, List[str]]:
    sequences: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"Line {line_no} in {path} has no items.")
            user, items = parts[0], parts[1:]
            if user in sequences:
                raise ValueError(f"Duplicated user [{user}] in {path}.")
            sequences[user] = items
    return sequences


def read_attributes(path: Path | None) -> Dict[str, List[str]]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item): [str(attr) for attr in attrs] for item, attrs in data.items()}


def write_inter(dataset_name: str, sequences: Dict[str, List[str]], out_dir: Path) -> None:
    with open(out_dir / f"{dataset_name}.inter", "w", encoding="utf-8", newline="") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for user in sorted(sequences, key=lambda x: int(x) if x.isdigit() else x):
            for pos, item in enumerate(sequences[user], start=1):
                f.write(f"{user}\t{item}\t1.0\t{pos}\n")


def write_item(dataset_name: str, items: Iterable[str], attrs: Dict[str, List[str]], out_dir: Path) -> None:
    with open(out_dir / f"{dataset_name}.item", "w", encoding="utf-8", newline="") as f:
        f.write("item_id:token\titem_attribute:token_seq\n")
        for item in sorted(items, key=lambda x: int(x) if x.isdigit() else x):
            attr_seq = " ".join(attrs.get(item, []))
            f.write(f"{item}\t{attr_seq}\n")


def write_stats(dataset_name: str, sequences: Dict[str, List[str]], out_dir: Path) -> None:
    lengths = [len(items) for items in sequences.values()]
    item_set: Set[str] = {item for items in sequences.values() for item in items}
    inter_num = sum(lengths)
    stats = {
        "dataset": dataset_name,
        "users": len(sequences),
        "items": len(item_set),
        "interactions": inter_num,
        "avg_len": inter_num / len(sequences) if sequences else 0.0,
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "sparsity_percent": (1.0 - inter_num / (len(sequences) * len(item_set))) * 100
        if sequences and item_set
        else 0.0,
    }
    with open(out_dir / f"{dataset_name}.stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def convert(seq_path: Path, attr_path: Path | None, dataset_name: str, out_dir: Path) -> None:
    sequences = read_sequences(seq_path)
    attrs = read_attributes(attr_path)
    items = {item for seq in sequences.values() for item in seq}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_inter(dataset_name, sequences, out_dir)
    write_item(dataset_name, items, attrs, out_dir)
    write_stats(dataset_name, sequences, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_path", required=True)
    parser.add_argument("--attr_path")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    convert(
        seq_path=Path(args.seq_path),
        attr_path=Path(args.attr_path) if args.attr_path else None,
        dataset_name=args.dataset_name,
        out_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
