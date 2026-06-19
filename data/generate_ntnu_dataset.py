#!/usr/bin/env python3
"""
Aggregate all organized JSONL datasets into a single training file.

Reads from the organized subdirectories (ntnu/, taiwan/) and combines
them into one comprehensive JSONL dataset. Also provides a convenience
function to regenerate all data using restructure_dataset.py.
"""

import json
from pathlib import Path
from typing import Dict, List


DATA_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> List[Dict]:
    """Load records from a JSONL file."""
    records = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def generate_dataset() -> List[Dict]:
    """Aggregate all organized datasets into one list.

    Reads from data/ntnu/ and data/taiwan/ subdirectories.

    Returns:
        List of all ChatML-format record dicts.
    """
    all_records = []

    sources = [
        DATA_DIR / "ntnu" / "general.jsonl",
        DATA_DIR / "ntnu" / "extras.jsonl",
        DATA_DIR / "taiwan" / "geography.jsonl",
        DATA_DIR / "taiwan" / "culture.jsonl",
        DATA_DIR / "taiwan" / "history.jsonl",
        DATA_DIR / "taiwan" / "politics.jsonl",
        DATA_DIR / "taiwan" / "economy.jsonl",
        DATA_DIR / "taiwan" / "universities.jsonl",
        DATA_DIR / "taiwan" / "districts.jsonl",
    ]

    for src in sources:
        records = load_jsonl(src)
        print(f"  Loaded {len(records):4d} records from {src.name}")
        all_records.extend(records)

    return all_records


def main():
    """Aggregate and write the combined dataset."""
    output_path = DATA_DIR / "full_dataset.jsonl"

    print("Aggregating organized datasets...")
    records = generate_dataset()
    print(f"\nTotal records: {len(records)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
