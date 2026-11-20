"""数据准备：解压/清洗 TinyStories、训练 BPE、打包 token id、构造 SFT 数据。"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_prompt, build_response  # noqa: E402
from tokenizer_bpe import (  # noqa: E402
    SPECIAL_TOKENS,
    load_tokenizer,
    special_token_ids,
    train_bpe,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _iter_items_from_json(data):
    if isinstance(data, dict):
        for key in ("text", "stories", "data", "content"):
            if key in data:
                data = data[key]
                break
        else:
            data = [data]
    if isinstance(data, str):
        data = [data]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                yield item.get("text") or item.get("story") or item.get("content") or ""


def iter_stories(tar_path: Path, max_stories: int, min_chars: int = 30, max_chars: int = 1200):
    """从 tar.gz 中流式读取故事文本，达到 max_stories 即停止。"""
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.lower()
            if name.endswith(".json"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = json.load(f)
                for text in _iter_items_from_json(data):
                    text = clean_text(text)
                    if min_chars <= len(text) <= max_chars:
                        yield text
                        count += 1
                        if count >= max_stories:
                            return
            elif name.endswith(".jsonl"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                for line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("story") or ""
                    else:
                        text = str(item)
                    text = clean_text(text)
                    if min_chars <= len(text) <= max_chars:
                        yield text
                        count += 1
                        if count >= max_stories:
                            return
            elif name.endswith(".txt"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read().decode("utf-8", errors="ignore")
                chunks = re.split(r"<\|endoftext\|>|\n\s*\n", raw)
                for text in chunks:
                    text = clean_text(text)
                    if min_chars <= len(text) <= max_chars:
                        yield text
                        count += 1
                        if count >= max_stories:
                            return


def load_or_build_stories(tar_path: Path, out_path: Path, max_stories: int, force: bool = False):
    if out_path.exists() and not force:
        texts = []
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                texts.append(item["text"])
        return texts

    print(f"从 {tar_path.name} 读取故事（最多 {max_stories} 条）...")
    texts = list(iter_stories(tar_path, max_stories))
    # 精确去重，保持顺序
    seen = set()
    unique = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    print(f"读取 {len(texts)} 条，去重后 {len(unique)} 条")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for text in unique:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return unique


def build_pretrain_data(
    texts: list[str],
    tokenizer_path: Path,
    vocab_size: int,
    tokenizer_chars: int,
    seed: int,
    force: bool = False,
):
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    n_val = max(200, int(len(texts) * 0.02))
    val_idx = set(indices[:n_val])
    train_texts = [texts[i] for i in range(len(texts)) if i not in val_idx]
    val_texts = [texts[i] for i in indices[:n_val]]

    if not tokenizer_path.exists() or force:
        print(f"训练 BPE 分词器（词表 {vocab_size}）...")
        sample, chars = [], 0
        for text in train_texts:
            sample.append(text)
            chars += len(text)
            if chars >= tokenizer_chars:
                break
        print(f"分词器训练语料：{len(sample)} 条，约 {chars/1e6:.1f}M 字符")
        train_bpe(sample, vocab_size=vocab_size, save_path=tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    print("词表大小:", tokenizer.get_vocab_size(), "eos_id:", eos_id)

    def encode_split(split_texts, bin_path: Path):
        if bin_path.exists() and not force:
            arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
            return len(arr)
        encodings = tokenizer.encode_batch(split_texts)
        total = sum(len(enc.ids) + 1 for enc in encodings)
        arr = np.empty(total, dtype=np.uint16)
        pos = 0
        for enc in encodings:
            ids = enc.ids
            arr[pos : pos + len(ids)] = np.asarray(ids, dtype=np.uint16)
            pos += len(ids)
            arr[pos] = eos_id
            pos += 1
        arr.tofile(bin_path)
        return total

    train_bin = DATA_DIR / "train.bin"
    val_bin = DATA_DIR / "val.bin"
    print("编码训练集...")
    n_train_tokens = encode_split(train_texts, train_bin)
    print("编码验证集...")
    n_val_tokens = encode_split(val_texts, val_bin)

    # 统计信息
    enc_lengths = [len(enc.ids) for enc in tokenizer.encode_batch(train_texts[:5000])]
    enc_lengths.sort()
    stats = {
        "num_stories_train": len(train_texts),
        "num_stories_val": len(val_texts),
        "train_chars": sum(len(t) for t in train_texts),
        "val_chars": sum(len(t) for t in val_texts),
        "train_tokens": int(n_train_tokens),
        "val_tokens": int(n_val_tokens),
        "tokens_per_char": n_train_tokens
        / max(1, sum(len(t) for t in train_texts)),
        "chars_per_token": sum(len(t) for t in train_texts) / max(1, n_train_tokens),
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos_id,
        "special_tokens": special_token_ids(tokenizer),
        "token_len_p50": enc_lengths[len(enc_lengths) // 2] if enc_lengths else 0,
        "token_len_p90": enc_lengths[int(len(enc_lengths) * 0.9)] if enc_lengths else 0,
        "token_len_max": enc_lengths[-1] if enc_lengths else 0,
    }
    (DATA_DIR / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return tokenizer, stats


def prepare_sentiment(tokenizer, seed: int = 42):
    import pandas as pd

    label_map = {1: "正面", 0: "负面"}
    instruction = "请判断下面这句话的情感倾向，只回答“正面”或“负面”。"

    def make_item(text: str, label: int) -> dict:
        text = text.strip()[:120]  # 控制长度，保证能放进 256 的上下文
        prompt = build_prompt(instruction + f"\n句子：{text}")
        response = build_response(label_map[int(label)])
        return {
            "prompt": prompt,
            "response": response,
            "text": text,
            "label": int(label),
            "label_text": label_map[int(label)],
        }

    splits = {}
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(DATA_DIR / f"senti_{split}.parquet")
        items = [make_item(row.text, row.label) for row in df.itertuples()]
        splits[split] = items
        out = DATA_DIR / f"sft_sentiment_{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"情感 {split}: {len(items)} 条 -> {out.name}")

    # 供评估用的 test 文件已经包含 gold label
    return splits


def prepare_instruct(tokenizer, seed: int = 42, max_train: int = 6000):
    path = DATA_DIR / "alpaca_gpt4_data_zh.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(data)

    items = []
    seen = set()
    for row in data:
        instruction = (row.get("instruction") or "").strip()
        input_text = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        if len(instruction) + len(input_text) + len(output) > 200:
            continue
        if len(output) > 110:
            continue
        key = (instruction, input_text, output)
        if key in seen:
            continue
        seen.add(key)
        prompt = build_prompt(instruction, input_text)
        response = build_response(output)
        items.append(
            {
                "prompt": prompt,
                "response": response,
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
        )

    n_val = 300
    val_items = items[:n_val]
    train_items = items[n_val : n_val + max_train]
    for name, subset in (("train", train_items), ("val", val_items)):
        out = DATA_DIR / f"sft_instruct_{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"指令 {name}: {len(subset)} 条 -> {out.name}")

    # 固定 20 条定性评估样例
    eval_items = val_items[:20]
    out = DATA_DIR / "sft_instruct_eval.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"指令评估样例: {len(eval_items)} 条 -> {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stories", type=int, default=30000)
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--tokenizer-chars", type=int, default=30_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stories_jsonl = DATA_DIR / "tinystories_zh.jsonl"
    tar_path = DATA_DIR / "TinyStories_all_data_zh_1M.tar.gz"
    texts = load_or_build_stories(tar_path, stories_jsonl, args.max_stories, args.force)

    if not args.skip_pretrain:
        tokenizer, stats = build_pretrain_data(
            texts,
            DATA_DIR / "tokenizer.json",
            args.vocab_size,
            args.tokenizer_chars,
            args.seed,
            args.force,
        )
    else:
        tokenizer = load_tokenizer(DATA_DIR / "tokenizer.json")

    if not args.skip_sft:
        prepare_sentiment(tokenizer, args.seed)
        prepare_instruct(tokenizer, args.seed)


if __name__ == "__main__":
    main()
