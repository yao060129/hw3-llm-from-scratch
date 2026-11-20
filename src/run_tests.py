"""模型与分词器的正确性测试，结果写入 results/logs/tests.json。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GPTConfig  # noqa: E402
from dataset import get_batch, load_memmap  # noqa: E402
from model import GPT, check_causality, check_kv_cache  # noqa: E402
from tokenizer_bpe import load_tokenizer, roundtrip_ok  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def overfit_test() -> dict:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(DATA / "tokenizer.json")
    config = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = GPT(config)
    data = load_memmap(DATA / "train.bin")
    x, y = get_batch(data, batch_size=4, block_size=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "steps": len(losses),
    }


def main() -> None:
    tokenizer = load_tokenizer(DATA / "tokenizer.json")
    texts = []
    with open(DATA / "tinystories_zh.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            texts.append(json.loads(line)["text"])

    results = {
        "causality": check_causality(),
        "kv_cache": check_kv_cache(),
        "tokenizer_roundtrip_200": roundtrip_ok(tokenizer, texts),
        "overfit": overfit_test(),
    }
    out = ROOT / "results" / "logs" / "tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
