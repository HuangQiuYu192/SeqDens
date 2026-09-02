#!/usr/bin/env python3
"""Prepare extra sequential recommendation datasets in RecBole atomic format.

The script supports two common inputs:
1. RecBole processed atomic zip/directories.
2. Common raw files for ML-1M, Yelp, and Amazon Books.

Outputs are written under ../<DatasetName>/ with .inter, .item and id maps,
matching the existing Beauty/Sports/Toys layout in this project.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import _utils as dutils


RAW_DIR = Path("raw")

DATASET_ALIASES = {
    "ml-1m": "ML-1M",
    "ML-1M": "ML-1M",
    "yelp": "Yelp",
    "Yelp": "Yelp",
    "yelp-2022": "Yelp",
    "yelp-2019": "Yelp-2019",
    "Yelp-2019": "Yelp-2019",
    "amazon-books": "Amazon-Books",
    "Amazon-Books": "Amazon-Books",
    "Books": "Amazon-Books",
}

RECBLE_RAW_NAMES = {
    "ML-1M": "ml-1m",
    "Yelp": "yelp-2022",
    "Yelp-2019": "yelp-2022",
    "Amazon-Books": "amazon-books",
}

AMAZON_REVIEW_NAMES = {
    "Amazon-Books": ["reviews_Books_5.json", "reviews_Books_5.json.gz"],
}

AMAZON_META_NAMES = {
    "Amazon-Books": ["meta_Books.json", "meta_Books.json.gz"],
}


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace('"', "'")
    return " ".join(text.split())


def normalize_dataset_name(name: str) -> str:
    if name not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset [{name}]. Use one of: {sorted(DATASET_ALIASES)}")
    return DATASET_ALIASES[name]


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def find_first(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_named_file(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    for name in names:
        matches = sorted(root.rglob(name)) if root.exists() else []
        if matches:
            return matches[0]
    return None


def ensure_output_dir(dataset_name: str) -> Path:
    out_dir = Path("..") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def copy_recbole_atomic(dataset_name: str, raw_dir: Path) -> bool:
    """Copy RecBole processed atomic files if a zip or directory is present."""
    recbole_name = RECBLE_RAW_NAMES[dataset_name]
    candidates = [
        raw_dir / f"{dataset_name}.zip",
        raw_dir / f"{recbole_name}.zip",
        raw_dir / dataset_name,
        raw_dir / recbole_name,
    ]
    source = find_first(candidates)
    if source is None:
        return False

    tmp_root = raw_dir / f".tmp_{dataset_name}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(tmp_root)
        search_root = tmp_root
    else:
        search_root = source

    atomic_files = []
    for suffix in (".inter", ".item", ".user"):
        atomic_files.extend(search_root.rglob(f"*{suffix}"))

    if not atomic_files:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        return False

    out_dir = ensure_output_dir(dataset_name)
    for src in atomic_files:
        suffix = src.suffix
        dst = out_dir / f"{dataset_name}{suffix}"
        shutil.copy2(src, dst)

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    print(f"[{dataset_name}] copied RecBole atomic files from {source}")
    return True


def filter_and_map(
    interactions: List[Tuple[str, str, float, int]],
    dataset_name: str,
    user_core: int,
    item_core: int,
):
    user_items, time_interval = dutils.get_interaction(
        [(u, i, t) for u, i, _, t in interactions], dataset_name
    )
    user_items, time_interval = dutils.filter_Kcore(
        user_items, time_interval, user_core=user_core, item_core=item_core
    )
    mapped_items, mapped_delta, user_num, item_num, data_maps = dutils.id_map(
        user_items, time_interval
    )
    kept_users = set(data_maps["user2id"].keys())
    kept_items = set(data_maps["item2id"].keys())
    kept_interactions = [
        (u, i, r, t) for u, i, r, t in interactions if u in kept_users and i in kept_items
    ]
    return mapped_items, mapped_delta, user_num, item_num, data_maps, kept_interactions


def write_outputs(
    dataset_name: str,
    interactions: List[Tuple[str, str, float, int]],
    item_features: Dict[str, Dict[str, str]],
    user_core: int,
    item_core: int,
):
    user_items, _, user_num, item_num, maps, kept_interactions = filter_and_map(
        interactions, dataset_name, user_core, item_core
    )
    out_dir = ensure_output_dir(dataset_name)
    user2id = maps["user2id"]
    item2id = maps["item2id"]
    id2user = maps["id2user"]
    id2item = maps["id2item"]

    with open(out_dir / f"{dataset_name}_user_id_map.json", "w", encoding="utf-8") as f:
        json.dump({"raw2id": user2id, "id2raw": id2user}, f, ensure_ascii=False, indent=2)

    with open(out_dir / f"{dataset_name}_item_id_map.json", "w", encoding="utf-8") as f:
        json.dump({"raw2id": item2id, "id2raw": id2item}, f, ensure_ascii=False, indent=2)

    with open(out_dir / f"{dataset_name}.inter", "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for raw_user, raw_item, rating, timestamp in kept_interactions:
            f.write(
                f"{user2id[raw_user]}\t{item2id[raw_item]}\t{float(rating)}\t{int(timestamp)}\n"
            )

    feature_names = sorted({k for fields in item_features.values() for k in fields.keys()})
    if not feature_names:
        feature_names = ["title"]
    with open(out_dir / f"{dataset_name}.item", "w", encoding="utf-8") as f:
        header = ["item_id:token"]
        for field in feature_names:
            field_type = "token_seq" if field in {"categories", "genres"} else "token"
            header.append(f"{field}:{field_type}")
        f.write("\t".join(header) + "\n")
        for numeric_id in sorted(id2item.keys(), key=lambda x: int(x)):
            raw_item = id2item[numeric_id]
            fields = item_features.get(raw_item, {})
            row = [numeric_id] + [clean_text(fields.get(field, "")) for field in feature_names]
            f.write("\t".join(row) + "\n")

    lengths = [len(seq) for seq in user_items.values()]
    print(
        f"[{dataset_name}] users={user_num}, items={item_num}, "
        f"inters={len(kept_interactions)}, avg_len={sum(lengths) / len(lengths):.2f}"
    )


def prepare_ml_1m(raw_dir: Path, user_core: int, item_core: int):
    base = find_first([raw_dir / "ml-1m", raw_dir / "ML-1M", raw_dir])
    ratings = find_named_file(base, ["ratings.dat", "ratings.csv"])
    movies = find_named_file(base, ["movies.dat", "movies.csv"])
    if ratings is None:
        raise FileNotFoundError("ML-1M requires ratings.dat or ratings.csv in raw/ml-1m/")

    interactions = []
    if ratings.name.endswith(".dat"):
        with open(ratings, "r", encoding="latin-1") as f:
            for line in f:
                user, item, rating, timestamp = line.rstrip("\n").split("::")
                interactions.append((user, item, float(rating), int(timestamp)))
    else:
        with open(ratings, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            idx = {name: i for i, name in enumerate(header)}
            for line in f:
                parts = line.rstrip("\n").split(",")
                interactions.append(
                    (
                        parts[idx["userId"]],
                        parts[idx["movieId"]],
                        float(parts[idx["rating"]]),
                        int(float(parts[idx["timestamp"]])),
                    )
                )

    item_features = {}
    if movies is not None and movies.name.endswith(".dat"):
        with open(movies, "r", encoding="latin-1") as f:
            for line in f:
                item, title, genres = line.rstrip("\n").split("::")
                item_features[item] = {
                    "title": title,
                    "genres": ", ".join(f"'{x}'" for x in genres.split("|") if x),
                }
    elif movies is not None:
        with open(movies, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            idx = {name: i for i, name in enumerate(header)}
            for line in f:
                parts = line.rstrip("\n").split(",")
                item_features[parts[idx["movieId"]]] = {
                    "title": parts[idx["title"]],
                    "genres": ", ".join(f"'{x}'" for x in parts[idx["genres"]].split("|") if x),
                }

    write_outputs("ML-1M", interactions, item_features, user_core, item_core)


def parse_json_line(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def prepare_amazon_books(raw_dir: Path, user_core: int, item_core: int):
    review = find_first([raw_dir / name for name in AMAZON_REVIEW_NAMES["Amazon-Books"]])
    meta = find_first([raw_dir / name for name in AMAZON_META_NAMES["Amazon-Books"]])
    if review is None:
        raise FileNotFoundError("Amazon-Books requires reviews_Books_5.json(.gz) in raw/")

    interactions = []
    with open_text(review) as f:
        for line in f:
            record = parse_json_line(line.strip())
            interactions.append(
                (
                    clean_text(record["reviewerID"]),
                    clean_text(record["asin"]),
                    float(record.get("overall", 1.0)),
                    int(record["unixReviewTime"]),
                )
            )

    item_features = {}
    if meta is not None:
        with open_text(meta) as f:
            for line in f:
                record = parse_json_line(line.strip())
                asin = clean_text(record.get("asin", ""))
                if not asin:
                    continue
                categories = []
                for path in record.get("categories", []) or []:
                    if isinstance(path, (list, tuple)):
                        categories.extend(clean_text(x) for x in path if clean_text(x))
                    elif clean_text(path):
                        categories.append(clean_text(path))
                item_features[asin] = {
                    "title": clean_text(record.get("title", "")),
                    "categories": ", ".join(f"'{x}'" for x in categories),
                }

    write_outputs("Amazon-Books", interactions, item_features, user_core, item_core)


def parse_start_timestamp(start_date: str | None) -> int | None:
    if not start_date:
        return None
    return int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())


def prepare_yelp(
    raw_dir: Path,
    dataset_name: str,
    user_core: int,
    item_core: int,
    min_rating: float,
    start_date: str | None,
):
    base = find_first([raw_dir / "yelp", raw_dir / "Yelp", raw_dir / "yelp-2022", raw_dir])
    review = find_named_file(
        base,
        [
            "yelp_academic_dataset_review.json",
            "review.json",
            "yelp_review.json",
            "yelp_academic_dataset_review.json.gz",
            "review.json.gz",
        ],
    )
    business = find_named_file(
        base,
        [
            "yelp_academic_dataset_business.json",
            "business.json",
            "yelp_business.json",
            "yelp_academic_dataset_business.json.gz",
            "business.json.gz",
        ],
    )
    if review is None:
        raise FileNotFoundError("Yelp requires yelp_academic_dataset_review.json(.gz) in raw/yelp/")

    interactions = []
    start_timestamp = parse_start_timestamp(start_date)
    with open_text(review) as f:
        for line in f:
            record = json.loads(line)
            rating = float(record.get("stars", 0.0))
            if rating < min_rating:
                continue
            timestamp = int(
                datetime.strptime(record["date"].split()[0], "%Y-%m-%d").timestamp()
            )
            if start_timestamp is not None and timestamp < start_timestamp:
                continue
            interactions.append(
                (
                    clean_text(record["user_id"]),
                    clean_text(record["business_id"]),
                    rating,
                    timestamp,
                )
            )

    item_features = {}
    if business is not None:
        with open_text(business) as f:
            for line in f:
                record = json.loads(line)
                bid = clean_text(record.get("business_id", ""))
                cats = [
                    clean_text(x)
                    for x in str(record.get("categories", "")).split(",")
                    if clean_text(x)
                ]
                item_features[bid] = {
                    "title": clean_text(record.get("name", "")),
                    "categories": ", ".join(f"'{x}'" for x in cats),
                }

    write_outputs(dataset_name, interactions, item_features, user_core, item_core)


def prepare_one(
    name: str,
    raw_dir: Path,
    user_core: int,
    item_core: int,
    yelp_min_rating: float,
    yelp_start_date: str | None,
):
    dataset_name = normalize_dataset_name(name)
    if copy_recbole_atomic(dataset_name, raw_dir):
        return
    if dataset_name == "ML-1M":
        prepare_ml_1m(raw_dir, user_core, item_core)
    elif dataset_name in {"Yelp", "Yelp-2019"}:
        prepare_yelp(
            raw_dir,
            dataset_name,
            user_core,
            item_core,
            yelp_min_rating,
            yelp_start_date,
        )
    elif dataset_name == "Amazon-Books":
        prepare_amazon_books(raw_dir, user_core, item_core)
    else:
        raise ValueError(dataset_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Datasets to prepare: ML-1M, Yelp, Amazon-Books, or all",
    )
    parser.add_argument("--raw_dir", default=str(RAW_DIR))
    parser.add_argument("--user_core", type=int, default=5)
    parser.add_argument("--item_core", type=int, default=5)
    parser.add_argument("--yelp_min_rating", type=float, default=4.0)
    parser.add_argument("--yelp_start_date", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    datasets = ["ML-1M", "Yelp", "Amazon-Books"] if "all" in args.datasets else args.datasets
    for dataset in datasets:
        prepare_one(
            dataset,
            raw_dir=raw_dir,
            user_core=args.user_core,
            item_core=args.item_core,
            yelp_min_rating=args.yelp_min_rating,
            yelp_start_date=args.yelp_start_date,
        )


if __name__ == "__main__":
    main()
