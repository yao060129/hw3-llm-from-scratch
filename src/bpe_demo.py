"""从零实现的字节/字符级 BPE（教学用，不用于正式训练）。

正式分词器由 `tokenizer_bpe.py` 使用 Rust 实现训练；本文件用纯 Python
展示 BPE 的核心：统计相邻符号对、合并最高频对、按合并顺序编码。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

END = "</w>"


def _word_to_symbols(word: str) -> tuple[str, ...]:
    """把一行文本拆成字符序列，并在末尾加上结束标记。"""
    return tuple(word) + (END,)


def train_bpe(texts: list[str], num_merges: int, min_frequency: int = 2):
    """训练 BPE：返回合并规则列表和词频表。"""
    word_freq: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        for line in text.splitlines():
            line = line.strip()
            if line:
                word_freq[_word_to_symbols(line)] += 1

    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pair_counts: Counter[tuple[str, str]] = Counter()
        for symbols, freq in word_freq.items():
            for a, b in zip(symbols, symbols[1:]):
                pair_counts[(a, b)] += freq
        if not pair_counts:
            break
        (best_a, best_b), best_count = pair_counts.most_common(1)[0]
        if best_count < min_frequency:
            break

        new_symbol = best_a + best_b
        merges.append((best_a, best_b))

        new_word_freq: Counter[tuple[str, ...]] = Counter()
        for symbols, freq in word_freq.items():
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == best_a
                    and symbols[i + 1] == best_b
                ):
                    merged.append(new_symbol)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            new_word_freq[tuple(merged)] += freq
        word_freq = new_word_freq

    return merges, word_freq


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """按训练得到的合并顺序编码一个词/行。"""
    symbols = list(word) + [END]
    for a, b in merges:
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


def decode_symbols(symbols: list[str]) -> str:
    return "".join(symbols).replace(END, "")


def demo(corpus_path: str | None, num_merges: int, max_lines: int, out_path: str) -> None:
    if corpus_path and Path(corpus_path).exists():
        raw_lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
        lines = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict) and "text" in item:
                    line = item["text"]
            except json.JSONDecodeError:
                pass
            lines.append(line)
            if len(lines) >= max_lines:
                break
        texts = ["\n".join(lines)]
    else:
        texts = [
            "从前有一个小女孩，她住在一个小村庄里。\n"
            "小女孩很喜欢小动物，她每天都会去森林里玩。\n"
            "有一天，她看见了一只受伤的小鸟。\n"
            "小女孩把小鸟带回家，给它喂水，包扎伤口。\n"
        ] * 100

    merges, word_freq = train_bpe(texts, num_merges)
    sample = texts[0].splitlines()[0]
    encoded = encode_word(sample, merges)
    decoded = decode_symbols(encoded)

    result = {
        "num_merges": len(merges),
        "merges": merges,
        "sample_text": sample,
        "sample_symbols": encoded,
        "decoded": decoded,
        "roundtrip_ok": decoded == sample,
        "unique_word_types": len(word_freq),
        "compression": {
            "chars": len(sample),
            "symbols": len(encoded),
            "chars_per_symbol": len(sample) / max(1, len(encoded)),
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "merges"},
                     ensure_ascii=False, indent=2))
    print("前 20 条合并规则:", merges[:20])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--num-merges", type=int, default=200)
    parser.add_argument("--max-lines", type=int, default=2000)
    parser.add_argument(
        "--out", type=str, default="results/logs/bpe_demo.json"
    )
    args = parser.parse_args()
    demo(args.corpus, args.num_merges, args.max_lines, args.out)
