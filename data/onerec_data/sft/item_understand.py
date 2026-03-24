"""
Item Understand Task
Input: caption parquet (pid, dense_caption) + pid2sid parquet
Output: LLM SFT training format parquet

Task: Given a video SID, generate its description/caption.
"""

import pandas as pd
import argparse
import json
import uuid
import random
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# ============== Configuration ==============
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'

# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一名视频描述生成器，请根据下面的视频token生成视频描述。",
    "你是一个专业的视频内容分析助手，能够理解视频token并生成准确的描述。",
    "你是一位视频理解专家，擅长将视频token转换为详细的文字描述。",
    "作为视频内容解析助手，你需要根据视频token提供精准的内容描述。",
    "你是一个智能视频解说员，可以根据视频token创建生动的描述。",
    "你具备理解视频token并生成高质量描述的能力。",
    "你是视频内容描述专家，能够将视频token转化为易懂的文字说明。",
    "作为AI视频分析助手，你可以根据视频token生成详细准确的描述。",
]

# User prompts (Chinese)
USER_PROMPTS = [
    "请描述 {sid} 的内容",
    "这段视频 {sid} 展示了什么？",
    "请解释 {sid} 中的内容",
    "能否说明 {sid} 里发生了什么？",
    "请分析 {sid} 的具体内容",
    "{sid} 这个视频讲的是什么？",
    "请详细描述 {sid}",
    "告诉我 {sid} 的内容是什么",
    "请为 {sid} 生成描述",
    "{sid} 包含哪些内容？",
    "请说明视频 {sid} 的主要内容",
    "描述一下 {sid} 中展现的场景",
    "{sid} 这段内容是关于什么的？",
    "请解读 {sid} 的视频内容",
    "能描述下 {sid} 吗？",
    "{sid} 里面有什么？",
    "请对 {sid} 进行内容说明",
    "这个 {sid} 是什么内容？",
    "分析 {sid} 并给出描述",
    "请阐述 {sid} 的内容细节",
]


# ============== Core Functions ==============
def pid_to_sid(pid, pid2sid: dict) -> str:
    """Convert a single pid to SID string."""
    if pid not in pid2sid:
        return ""
    code = pid2sid[pid]
    return SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])


def build_messages(sid: str, caption: str) -> str:
    """Build messages format JSON string."""
    system_prompt = random.choice(SYSTEM_PROMPTS)
    user_prompt = random.choice(USER_PROMPTS).format(sid=sid)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": caption}]}
    ]
    return json.dumps(messages, ensure_ascii=False)


def process_row(row, pid2sid: dict) -> dict:
    """Process a single row of data."""
    pid = row['pid']
    dense_caption = row['dense_caption']

    # Check data validity
    if dense_caption is None or (isinstance(dense_caption, float) and pd.isna(dense_caption)):
        return None
    if not dense_caption:
        return None

    # Convert pid to SID
    sid = pid_to_sid(pid, pid2sid)
    if not sid:
        return None

    return {
        'source': 'RecIF_ItemUnderstand',
        'uuid': str(uuid.uuid4()),
        'messages': build_messages(sid, dense_caption),
        'metadata': json.dumps({'pid': int(pid), 'sid': sid}, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Item Understand Task Data Processing")
    parser.add_argument('--input', type=str, required=True, help='Input caption parquet path')
    parser.add_argument('--pid2sid', type=str, required=True, help='pid2sid mapping parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=100000, help='Rows per parquet batch')
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pid2sid mapping
    print(f"Loading pid2sid from {args.pid2sid}...")
    df_pid2sid = pd.read_parquet(args.pid2sid)
    pid2sid = dict(zip(df_pid2sid['pid'], df_pid2sid['sid']))
    print(f"  Loaded {len(pid2sid):,} mappings")

    output_path = output_dir / 'train.parquet'

    # 2. Stream caption data and write parquet incrementally to avoid OOM on full pid2caption.
    print(f"Loading caption data from {args.input}...")
    parquet_file = pq.ParquetFile(args.input)
    total_rows = parquet_file.metadata.num_rows
    print(f"  Loaded metadata for {total_rows:,} rows")

    print("Processing...")
    writer = None
    saved_rows = 0

    try:
        with tqdm(total=total_rows) as pbar:
            for batch in parquet_file.iter_batches(
                batch_size=args.batch_size,
                columns=['pid', 'dense_caption'],
            ):
                df_batch = batch.to_pandas()
                results = []
                for _, row in df_batch.iterrows():
                    result = process_row(row, pid2sid)
                    if result:
                        results.append(result)

                if results:
                    table = pa.Table.from_pylist(results)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema)
                    writer.write_table(table)
                    saved_rows += len(results)

                pbar.update(len(df_batch))
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pd.DataFrame(columns=['source', 'uuid', 'messages', 'metadata']).to_parquet(output_path, index=False)

    print(f"Saved: {output_path} ({saved_rows:,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
