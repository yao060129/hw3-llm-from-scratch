"""字节级 BPE 分词器的训练与加载。

训练使用 HuggingFace `tokenizers` 的 Rust 实现（速度快），
分词算法本身在 `bpe_demo.py` 中有一个从零实现的教学版本。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>"]


def train_bpe(
    texts: list[str],
    vocab_size: int,
    save_path: str | Path,
    min_frequency: int = 2,
) -> Tokenizer:
    """在给定语料上训练 byte-level BPE 分词器并保存为 JSON。"""
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode(tokenizer: Tokenizer, text: str, add_special_tokens: bool = False) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def decode(tokenizer: Tokenizer, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)


def special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    return {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}


def roundtrip_ok(tokenizer: Tokenizer, samples: list[str]) -> bool:
    """检查 encode->decode 是否能还原文本。"""
    for text in samples:
        ids = tokenizer.encode(text).ids
        if tokenizer.decode(ids, skip_special_tokens=True) != text:
            return False
    return True


def corpus_stats(
    tokenizer: Tokenizer,
    texts: list[str],
    max_texts: int | None = None,
) -> dict:
    """统计语料的字符数、token 数、压缩率与长度分布。"""
    if max_texts is not None:
        texts = texts[:max_texts]

    total_chars = 0
    total_tokens = 0
    lengths: list[int] = []
    token_counter: Counter[int] = Counter()

    for text in texts:
        ids = tokenizer.encode(text).ids
        total_chars += len(text)
        total_tokens += len(ids)
        lengths.append(len(ids))
        token_counter.update(ids)

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def percentile(p: float) -> int:
        if n == 0:
            return 0
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return lengths_sorted[idx]

    top_tokens = []
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, count in token_counter.most_common(20):
        top_tokens.append(
            {
                "id": token_id,
                "token": id_to_token.get(token_id, ""),
                "count": count,
            }
        )

    return {
        "num_texts": n,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "tokens_per_char": total_tokens / max(1, total_chars),
        "chars_per_token": total_chars / max(1, total_tokens),
        "length_min": lengths_sorted[0] if n else 0,
        "length_p50": percentile(0.50),
        "length_p90": percentile(0.90),
        "length_p99": percentile(0.99),
        "length_max": lengths_sorted[-1] if n else 0,
        "top_tokens": top_tokens,
        "vocab_size": tokenizer.get_vocab_size(),
    }


if __name__ == "__main__":
    # 简单自检：训练一个很小的分词器并做 round-trip
    sample_texts = [
        "从前，有一个小女孩住在一个小村庄里。",
        "小猫看见了一只小鸟，它很想和小鸟一起玩。",
        "The quick brown fox jumps over the lazy dog.",
    ] * 200
    tok = train_bpe(sample_texts, vocab_size=512, save_path="/tmp/bpe_test.json")
    print("词表大小:", tok.get_vocab_size())
    print("round-trip:", roundtrip_ok(tok, sample_texts[:3]))
    stats = corpus_stats(tok, sample_texts[:10])
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:500])
