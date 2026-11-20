"""交互式演示：故事续写、情感分类、指令回答。

用法（在“作业三”目录下）：
    python src/demo.py story --prompt "从前，有一个小女孩"
    python src/demo.py sentiment --text "这家酒店的服务非常好"
    python src/demo.py instruct --instruction "把下面这句话翻译成英文" --input "你好"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GPTConfig  # noqa: E402
from dataset import build_prompt  # noqa: E402
from lora import apply_lora  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer_bpe import load_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPTS = {
    "story": ROOT / "results/checkpoints/pretrain/ckpt_best.pt",
    "sentiment": ROOT / "results/checkpoints/sft_sentiment_full/ckpt_best.pt",
    "instruct": ROOT / "results/checkpoints/sft_instruct_full/ckpt_best.pt",
}


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    method = ckpt.get("method", "full")
    if method == "lora":
        apply_lora(model, r=ckpt.get("lora_r", 8), alpha=ckpt.get("lora_alpha", 16))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["story", "sentiment", "instruct"])
    parser.add_argument("--prompt", type=str, default="从前，有一个小女孩")
    parser.add_argument("--text", type=str, default="这家酒店的服务非常好，下次还会再来。")
    parser.add_argument("--instruction", type=str, default="把下面这句话翻译成英文")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ckpt_path = Path(args.ckpt) if args.ckpt else CKPTS[args.mode]
    model, config, ckpt = load_model(ckpt_path)
    tokenizer = load_tokenizer(ROOT / "data/tokenizer.json")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    if args.mode == "story":
        prompt = args.prompt
        max_new = args.max_new_tokens or 80
        temperature = args.temperature if args.temperature is not None else 0.8
        ids = tokenizer.encode(prompt).ids
        out = model.generate(
            torch.tensor([ids]), max_new_tokens=max_new,
            temperature=temperature, top_k=args.top_k, eos_id=eos_id,
        )
        print(tokenizer.decode(out[0].tolist()))

    elif args.mode == "sentiment":
        instruction = "请判断下面这句话的情感倾向，只回答“正面”或“负面”。"
        prompt = build_prompt(instruction + f"\n句子：{args.text.strip()[:120]}")
        ids = tokenizer.encode(prompt).ids
        out = model.generate(
            torch.tensor([ids]), max_new_tokens=6,
            temperature=0.0, eos_id=eos_id,
        )
        generated = tokenizer.decode(out[0, len(ids) :].tolist())
        print(f"输入：{args.text}")
        print(f"模型输出：{generated!r}")
        print(f"预测：{'正面' if '正面' in generated[:12] else ('负面' if '负面' in generated[:12] else '无法判断')}")

    else:
        prompt = build_prompt(args.instruction, args.input)
        ids = tokenizer.encode(prompt).ids
        max_new = args.max_new_tokens or 48
        temperature = args.temperature if args.temperature is not None else 0.7
        out = model.generate(
            torch.tensor([ids]), max_new_tokens=max_new,
            temperature=temperature, top_k=args.top_k, eos_id=eos_id,
        )
        print("指令：", args.instruction)
        if args.input:
            print("输入：", args.input)
        print("回答：", tokenizer.decode(out[0, len(ids) :].tolist()))


if __name__ == "__main__":
    main()
