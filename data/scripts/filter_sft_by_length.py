#!/usr/bin/env python3
"""Analyze and filter SFT parquet shards by tokenized sequence length."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file-list", required=True, help="Path to split_data_sft/file_list.json")
    parser.add_argument("--output-dir", required=True, help="Directory to store filtered shards and reports")
    parser.add_argument("--base-model-dir", required=True, help="Tokenizer/model directory")
    parser.add_argument("--max-length", type=int, required=True, help="Maximum allowed token length")
    parser.add_argument("--rows-per-shard", type=int, default=1000, help="Rows per output parquet shard")
    parser.add_argument("--batch-size", type=int, default=256, help="Parquet read batch size")
    return parser.parse_args()


def load_file_list(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        file_list = json.load(fh)
    if not isinstance(file_list, list):
        raise ValueError(f"{path} must contain a JSON list of parquet files")
    return [str(Path(item)) for item in file_list]


def maybe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict) and content.get("type") == "text":
            text = content["text"]
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
            text = "".join(parts)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
        converted.append({"role": msg["role"], "content": text})
    return converted


def build_text_from_row(row: dict[str, Any], tokenizer) -> tuple[str, str]:
    source_name = row.get("source") or row.get("source_name") or "unknown"
    messages = row.get("message") or row.get("messages")
    segments = row.get("segments")

    if messages is not None:
        parsed_messages = maybe_json_loads(messages)
        text = tokenizer.apply_chat_template(
            convert_messages(parsed_messages),
            tokenize=False,
            add_generation_prompt=False,
        )
        return source_name, text + tokenizer.pad_token

    if segments is not None:
        parsed_segments = maybe_json_loads(segments)
        text = "".join(segment["text"] for segment in parsed_segments if segment.get("type") == "text")
        return source_name, text + tokenizer.pad_token

    raise ValueError("Row must contain messages/message or segments")


def tokenize_lengths(texts: list[str], tokenizer) -> list[int]:
    encoded = tokenizer(
        texts,
        padding=False,
        truncation=False,
    )
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def flush_rows(rows: list[dict[str, Any]], output_dir: Path, shard_idx: int, num_digits: int = 5) -> Path:
    output_path = output_dir / f"part-{shard_idx:0{num_digits}d}.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False, compression="snappy")
    return output_path


def main() -> None:
    args = parse_args()
    input_file_list = Path(args.input_file_list)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    parquet_files = load_file_list(input_file_list)

    source_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "kept": 0, "dropped": 0, "max_len": 0})
    overall = {"total": 0, "kept": 0, "dropped": 0, "max_len": 0}

    output_rows: list[dict[str, Any]] = []
    output_files: list[str] = []
    shard_idx = 0

    for parquet_file in parquet_files:
        parquet_reader = pq.ParquetFile(parquet_file)
        for batch in parquet_reader.iter_batches(batch_size=args.batch_size):
            batch_df = batch.to_pandas()
            rows = batch_df.to_dict("records")
            batch_items: list[tuple[dict[str, Any], str, str]] = []
            for row in rows:
                source_name, text = build_text_from_row(row, tokenizer)
                batch_items.append((row, source_name, text))

            lengths = tokenize_lengths([item[2] for item in batch_items], tokenizer)

            for (row, source_name, _text), length in zip(batch_items, lengths, strict=True):

                source_stats[source_name]["total"] += 1
                source_stats[source_name]["max_len"] = max(source_stats[source_name]["max_len"], length)
                overall["total"] += 1
                overall["max_len"] = max(overall["max_len"], length)

                if length > args.max_length:
                    source_stats[source_name]["dropped"] += 1
                    overall["dropped"] += 1
                    continue

                source_stats[source_name]["kept"] += 1
                overall["kept"] += 1
                output_rows.append(row)

                if len(output_rows) >= args.rows_per_shard:
                    output_path = flush_rows(output_rows, output_dir, shard_idx)
                    output_files.append(str(output_path.resolve()))
                    output_rows = []
                    shard_idx += 1

    if output_rows:
        output_path = flush_rows(output_rows, output_dir, shard_idx)
        output_files.append(str(output_path.resolve()))

    with (output_dir / "file_list.json").open("w", encoding="utf-8") as fh:
        json.dump(output_files, fh, indent=2, ensure_ascii=False)

    stats_payload = {
        "input_file_list": str(input_file_list.resolve()),
        "output_dir": str(output_dir.resolve()),
        "max_length": args.max_length,
        "overall": overall,
        "sources": {
            source: {
                **stats,
                "drop_ratio": (stats["dropped"] / stats["total"]) if stats["total"] else 0.0,
                "keep_ratio": (stats["kept"] / stats["total"]) if stats["total"] else 0.0,
            }
            for source, stats in sorted(source_stats.items())
        },
    }
    with (output_dir / "length_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats_payload, fh, indent=2, ensure_ascii=False)

    stats_rows = []
    for source_name, stats in sorted(source_stats.items()):
        total = stats["total"]
        stats_rows.append(
            {
                "source": source_name,
                "total": total,
                "kept": stats["kept"],
                "dropped": stats["dropped"],
                "keep_ratio": (stats["kept"] / total) if total else 0.0,
                "drop_ratio": (stats["dropped"] / total) if total else 0.0,
                "max_len": stats["max_len"],
            }
        )
    pd.DataFrame(stats_rows).to_csv(output_dir / "length_stats.csv", index=False)

    print(json.dumps(stats_payload["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
