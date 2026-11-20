"""评估与生成脚本：困惑度、情感分类、指令样例、自由生成。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GPTConfig  # noqa: E402
from dataset import get_batch, load_memmap, read_jsonl  # noqa: E402
from lora import apply_lora  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer_bpe import load_tokenizer  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--task", type=str, default="ppl", choices=["ppl", "sentiment", "instruct", "generate"]
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test", "validation"]
    )
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_any_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    method = ckpt.get("method", "full")
    if method == "lora":
        apply_lora(
            model,
            r=ckpt.get("lora_r", 8),
            alpha=ckpt.get("lora_alpha", 16),
            dropout=0.0,
        )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    tokenizer = load_tokenizer(Path(args.data_dir) / "tokenizer.json")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    model, config, ckpt = load_any_model(args.ckpt)
    method = ckpt.get("method", "full")
    name = args.name or Path(args.ckpt).parent.name + "_" + method
    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    print(f"模型: {args.ckpt} | method={method} | 参数 {model.num_parameters()/1e6:.3f}M")

    if args.task == "ppl":
        data = load_memmap(Path(args.data_dir) / f"{args.split}.bin")
        losses = []
        for _ in range(args.max_batches):
            x, y = get_batch(data, args.batch_size, config.block_size)
            with torch.no_grad():
                _, loss, _, _ = model(x, targets=y)
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        result = {
            "checkpoint": args.ckpt,
            "split": args.split,
            "loss": mean_loss,
            "perplexity": math.exp(min(20.0, mean_loss)),
            "num_batches": args.max_batches,
        }
        path = out_dir / "logs" / f"eval_{name}_ppl.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.task == "sentiment":
        from train_sft import compute_sentiment_metrics, predict_sentiment

        split = "test" if args.split == "test" else args.split
        path_map = {
            "train": "sft_sentiment_train.jsonl",
            "val": "sft_sentiment_validation.jsonl",
            "test": "sft_sentiment_test.jsonl",
            "validation": "sft_sentiment_validation.jsonl",
        }
        items = read_jsonl(Path(args.data_dir) / path_map[split])
        if args.max_items:
            items = items[: args.max_items]
        preds, labels, outputs = predict_sentiment(model, tokenizer, items, eos_id)
        metrics = compute_sentiment_metrics(preds, labels)
        metrics.update({"checkpoint": args.ckpt, "split": split, "method": method, "name": name})
        path = out_dir / "logs" / f"eval_{name}_sentiment_{split}.json"
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        pred_path = out_dir / "samples" / f"eval_{name}_sentiment_{split}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in outputs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    elif args.task == "instruct":
        from train_sft import generate_instruct_samples

        items = read_jsonl(Path(args.data_dir) / "sft_instruct_eval.jsonl")
        if args.max_items:
            items = items[: args.max_items]
        results = generate_instruct_samples(
            model, tokenizer, items, eos_id, max_new_tokens=args.max_new_tokens
        )
        path = out_dir / "samples" / f"eval_{name}_instruct.json"
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        for row in results[:3]:
            print("-" * 60)
            print("指令:", row["instruction"])
            print("参考:", row["reference"][:120])
            print("生成:", row["generated"][:200])

    else:  # generate
        prompts = args.prompt or ["从前，有一个小女孩", "一天，小明和小红一起去公园"]
        results = []
        for prompt in prompts:
            ids = tokenizer.encode(prompt).ids
            x = torch.tensor([ids], dtype=torch.long)
            out = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_id=eos_id,
            )
            text = tokenizer.decode(out[0].tolist())
            results.append({"prompt": prompt, "text": text})
            print("-" * 60)
            print(text)
        path = out_dir / "samples" / f"eval_{name}_generate.json"
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
