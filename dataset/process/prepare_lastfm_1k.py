#!/usr/bin/env python3
"""Prepare LastFM-1K in RecBole atomic format with only 5-core filtering."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


RAW_MEMBER = "lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
csv.field_size_limit(1024 * 1024 * 1024)


Interaction = Tuple[str, str, int]
ItemFeature = Dict[str, str]


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace('"', "'")
    return " ".join(text.split())


def parse_timestamp(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


def normalize_name(value: str) -> str:
    return clean_text(value).casefold()


def build_item_id(
    artist_id: str,
    artist_name: str,
    track_id: str,
    track_name: str,
    item_level: str,
) -> str | None:
    artist_id = clean_text(artist_id)
    artist_name = clean_text(artist_name)
    track_id = clean_text(track_id)
    track_name = clean_text(track_name)

    if item_level == "artist":
        if artist_id:
            return f"artist::{artist_id}"
        if artist_name:
            return f"artist_name::{normalize_name(artist_name)}"
        return None

    if track_id:
        return f"track::{track_id}"
    if track_name and (artist_id or artist_name):
        artist_part = artist_id if artist_id else normalize_name(artist_name)
        return f"track_name::{artist_part}::{normalize_name(track_name)}"
    if track_name:
        return f"track_name::{normalize_name(track_name)}"
    return None


def open_lastfm_rows(path: Path) -> Iterable[List[str]]:
    if path.suffixes[-2:] == [".tar", ".gz"] or path.name.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as archive:
            member = archive.extractfile(RAW_MEMBER)
            if member is None:
                raise FileNotFoundError(f"{RAW_MEMBER} not found in {path}")
            with gzip.open(member, mode="rt", encoding="utf-8", errors="replace") if False else member as raw:
                text = (line.decode("utf-8", errors="replace") for line in raw)
                yield from csv.reader(text, delimiter="\t")
    else:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            yield from csv.reader(f, delimiter="\t")


def read_interactions(path: Path, item_level: str) -> Tuple[Dict[str, List[Interaction]], Dict[str, ItemFeature], int]:
    user_seq: Dict[str, List[Interaction]] = defaultdict(list)
    item_features: Dict[str, ItemFeature] = {}
    skipped = 0

    for row in open_lastfm_rows(path):
        if len(row) < 6:
            skipped += 1
            continue
        user_id, timestamp, artist_id, artist_name, track_id, track_name = row[:6]
        item_id = build_item_id(artist_id, artist_name, track_id, track_name, item_level)
        if not user_id or item_id is None:
            skipped += 1
            continue
        try:
            unix_time = parse_timestamp(timestamp)
        except ValueError:
            skipped += 1
            continue

        user_seq[user_id].append((user_id, item_id, unix_time))
        if item_id not in item_features:
            item_features[item_id] = {
                "artist_id": clean_text(artist_id),
                "artist_name": clean_text(artist_name),
                "track_id": clean_text(track_id),
                "track_name": clean_text(track_name),
            }

    return user_seq, item_features, skipped


def kcore_filter(
    user_seq: Dict[str, List[Interaction]],
    user_core: int,
    item_core: int,
) -> Dict[str, List[Interaction]]:
    while True:
        item_count = Counter()
        for interactions in user_seq.values():
            item_count.update(item for _, item, _ in interactions)

        removed_any = False
        kept_users: Dict[str, List[Interaction]] = {}
        for user, interactions in user_seq.items():
            kept = [record for record in interactions if item_count[record[1]] >= item_core]
            if len(kept) >= user_core:
                kept_users[user] = kept
            else:
                removed_any = True
            if len(kept) != len(interactions):
                removed_any = True

        if not removed_any:
            return kept_users
        user_seq = kept_users


def write_atomic(
    dataset_name: str,
    user_seq: Dict[str, List[Interaction]],
    item_features: Dict[str, ItemFeature],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    user2id: Dict[str, str] = {}
    item2id: Dict[str, str] = {}
    id2user: Dict[str, str] = {}
    id2item: Dict[str, str] = {}

    ordered_records: List[Interaction] = []
    for raw_user in sorted(user_seq):
        records = sorted(user_seq[raw_user], key=lambda x: x[2])
        uid = str(len(user2id) + 1)
        user2id[raw_user] = uid
        id2user[uid] = raw_user
        for _, raw_item, timestamp in records:
            if raw_item not in item2id:
                iid = str(len(item2id) + 1)
                item2id[raw_item] = iid
                id2item[iid] = raw_item
            ordered_records.append((raw_user, raw_item, timestamp))

    with open(out_dir / f"{dataset_name}.inter", "w", encoding="utf-8", newline="") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for raw_user, raw_item, timestamp in ordered_records:
            f.write(f"{user2id[raw_user]}\t{item2id[raw_item]}\t1.0\t{timestamp}\n")

    with open(out_dir / f"{dataset_name}.item", "w", encoding="utf-8", newline="") as f:
        f.write("item_id:token\tartist_id:token\tartist_name:token\ttrack_id:token\ttrack_name:token\n")
        for numeric_id in sorted(id2item, key=lambda x: int(x)):
            raw_item = id2item[numeric_id]
            fields = item_features.get(raw_item, {})
            row = [
                numeric_id,
                clean_text(fields.get("artist_id", "")),
                clean_text(fields.get("artist_name", "")),
                clean_text(fields.get("track_id", "")),
                clean_text(fields.get("track_name", "")),
            ]
            f.write("\t".join(row) + "\n")

    with open(out_dir / f"{dataset_name}_user_id_map.json", "w", encoding="utf-8") as f:
        json.dump({"raw2id": user2id, "id2raw": id2user}, f, ensure_ascii=False, indent=2)
    with open(out_dir / f"{dataset_name}_item_id_map.json", "w", encoding="utf-8") as f:
        json.dump({"raw2id": item2id, "id2raw": id2item}, f, ensure_ascii=False, indent=2)

    lengths = [len(records) for records in user_seq.values()]
    stats = {
        "dataset": dataset_name,
        "users": len(user2id),
        "items": len(item2id),
        "interactions": len(ordered_records),
        "avg_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
    }
    with open(out_dir / f"{dataset_name}.stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="raw/LastFM/lastfm-dataset-1K.tar.gz",
        help="Path to LastFM-1K tar.gz or extracted interaction TSV.",
    )
    parser.add_argument("--dataset_name", default="LastFM")
    parser.add_argument("--output_dir", default="../LastFM")
    parser.add_argument("--user_core", type=int, default=5)
    parser.add_argument("--item_core", type=int, default=5)
    parser.add_argument("--item_level", choices=["track", "artist"], default="track")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    user_seq, item_features, skipped = read_interactions(input_path, args.item_level)
    raw_users = len(user_seq)
    raw_interactions = sum(len(records) for records in user_seq.values())
    raw_items = len(item_features)
    print(
        f"[raw] users={raw_users}, items={raw_items}, "
        f"interactions={raw_interactions}, skipped={skipped}"
    )

    filtered = kcore_filter(user_seq, args.user_core, args.item_core)
    kept_items = {item for records in filtered.values() for _, item, _ in records}
    item_features = {item: item_features[item] for item in kept_items if item in item_features}
    write_atomic(args.dataset_name, filtered, item_features, output_dir)


if __name__ == "__main__":
    main()
