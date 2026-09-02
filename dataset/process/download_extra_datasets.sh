#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$ROOT/raw"
mkdir -p "$RAW_DIR"

download() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    echo "[skip] $out"
  else
    echo "[download] $url"
    wget -c "$url" -O "$out"
  fi
}

case "${1:-all-small}" in
  ml-1m)
    mkdir -p "$RAW_DIR/ml-1m"
    download "https://files.grouplens.org/datasets/movielens/ml-1m.zip" "$RAW_DIR/ml-1m/ml-1m.zip"
    unzip -o -q "$RAW_DIR/ml-1m/ml-1m.zip" -d "$RAW_DIR/ml-1m"
    ;;
  amazon-books)
    download "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz" "$RAW_DIR/reviews_Books_5.json.gz"
    download "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz" "$RAW_DIR/meta_Books.json.gz"
    ;;
  yelp)
    cat <<'MSG'
Yelp raw data requires an authorized local copy.
Put either a RecBole atomic zip/dir or Yelp official JSON files under dataset/process/raw:
  raw/Yelp.zip or raw/yelp-2022.zip
  raw/yelp/yelp_academic_dataset_review.json
  raw/yelp/yelp_academic_dataset_business.json
Then run:
  python prepare_extra_datasets.py Yelp
MSG
    ;;
  all-small)
    "$0" ml-1m
    "$0" yelp
    ;;
  all)
    "$0" ml-1m
    "$0" amazon-books
    "$0" yelp
    ;;
  *)
    echo "Usage: $0 {ml-1m|amazon-books|yelp|all-small|all}" >&2
    exit 1
    ;;
esac
