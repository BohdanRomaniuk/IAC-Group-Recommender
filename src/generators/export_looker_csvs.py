"""
Convert MovieLens group data (.dat files) to CSVs ready for BigQuery / Looker Studio.

Usage (from src/):
    python -m generators.export_looker_csvs                        # 32m dataset (default)
    python -m generators.export_looker_csvs --dataset 1m           # 1m dataset
    python -m generators.export_looker_csvs --out /tmp/looker_csvs # custom output dir

Output tables
-------------
groups.csv         group_id, member_count
group_members.csv  group_id, user_id
group_ratings.csv  group_id, movie_id, rating, timestamp, split
user_ratings.csv   user_id,  movie_id, rating, timestamp, split
users.csv          user_id
movies.csv         movie_id, title, year, genres  (pipe-separated genres, one row per movie)
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path



BASE_DIR = Path(__file__).parent  # src/generators/


def parse_args():
    p = argparse.ArgumentParser(description="Export .dat files to Looker-ready CSVs")
    p.add_argument("--dataset", choices=["1m", "32m"], default="32m")
    p.add_argument("--out", default=None, help="Output directory (default: data/<dataset>/looker_csvs/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_group_members(path: Path):
    """Yield (group_id, user_id) pairs from groupMember.dat."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group_id, members_str = line.split(" ", 1)
            for user_id in members_str.split(","):
                yield int(group_id), int(user_id.strip())


def parse_ratings(path: Path, split: str):
    """Yield (entity_id, movie_id, rating, timestamp_iso, split) rows.
    Format: entity_id movie_id rating timestamp
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            entity_id = int(parts[0])
            movie_id  = int(parts[1])
            rating    = int(parts[2])
            ts        = int(parts[3])
            ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            yield entity_id, movie_id, rating, ts_iso, split


def parse_users(path: Path):
    """Yield (user_id,) from users.dat. Demographic fields are dummy data and excluded."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            try:
                user_id = int(parts[0])
            except ValueError:
                continue
            yield (user_id,)


def _extract_year(title: str):
    """Pull the (YYYY) year out of a movie title, return ('clean title', year_or_None)."""
    m = re.search(r"\((\d{4})\)\s*$", title)
    if m:
        return title[: m.start()].strip(), int(m.group(1))
    return title.strip(), None


def parse_movies(path: Path):
    """Yield (movie_id, title, year, genres) rows — one row per movie."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 3:
                continue
            # Strip non-numeric prefix from movie_id (guards against stray BOM chars)
            raw_id = re.sub(r"[^\d]", "", parts[0])
            if not raw_id:
                continue
            movie_id = int(raw_id)
            title_raw = parts[1].strip()
            genres_str = parts[2].strip()
            title, year = _extract_year(title_raw)
            yield movie_id, title, year, genres_str


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, header: list, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def export(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- group_members.csv & groups.csv ---
    print("Parsing groupMember.dat …")
    member_rows = list(parse_group_members(data_dir / "groupMember.dat"))
    write_csv(out_dir / "group_members.csv", ["group_id", "user_id"], member_rows)

    from collections import Counter
    member_counts = Counter(gid for gid, _ in member_rows)
    group_rows = [(gid, cnt) for gid, cnt in sorted(member_counts.items())]
    write_csv(out_dir / "groups.csv", ["group_id", "member_count"], group_rows)

    # --- group_ratings.csv ---
    print("Parsing group rating files …")
    group_rating_rows = []
    for fname, split in [
        ("groupRatingTrain.dat", "train"),
        ("groupRatingVal.dat",   "val"),
        ("groupRatingTest.dat",  "test"),
    ]:
        p = data_dir / fname
        if p.exists():
            group_rating_rows.extend(parse_ratings(p, split))
    write_csv(out_dir / "group_ratings.csv",
              ["group_id", "movie_id", "rating", "timestamp", "split"],
              group_rating_rows)

    # --- user_ratings.csv ---
    print("Parsing user rating files …")
    user_rating_rows = []
    for fname, split in [
        ("userRatingTrain.dat", "train"),
        ("userRatingVal.dat",   "val"),
        ("userRatingTest.dat",  "test"),
    ]:
        p = data_dir / fname
        if p.exists():
            user_rating_rows.extend(parse_ratings(p, split))
    write_csv(out_dir / "user_ratings.csv",
              ["user_id", "movie_id", "rating", "timestamp", "split"],
              user_rating_rows)

    # --- users.csv ---
    print("Parsing users.dat …")
    user_rows = list(parse_users(data_dir / "users.dat"))
    write_csv(out_dir / "users.csv", ["user_id"], user_rows)

    # --- movies.csv ---
    print("Parsing movies.dat …")
    movie_rows = list(parse_movies(data_dir / "movies.dat"))
    write_csv(out_dir / "movies.csv",
              ["movie_id", "title", "year", "genres"],
              movie_rows)

    # Summary
    print("\nDone! Files written to:", out_dir)
    for csv_file in sorted(out_dir.glob("*.csv")):
        size_kb = csv_file.stat().st_size // 1024
        with open(csv_file, encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1  # minus header
        print(f"  {csv_file.name:<30} {rows:>8,} rows   {size_kb:>6} KB")


def main():
    args = parse_args()
    data_dir = BASE_DIR / "data" / f"MovieLens-{args.dataset}"
    if not data_dir.exists():
        sys.exit(f"Data directory not found: {data_dir}\n"
                 f"Run `python -m generators.generator_{args.dataset}` first.")
    out_dir = Path(args.out) if args.out else data_dir / "looker_csvs"
    export(data_dir, out_dir)


if __name__ == "__main__":
    main()
