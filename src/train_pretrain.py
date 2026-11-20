"""GPT 预训练主程序（CPU 友好）。

用法示例（在 作业三 目录下）：
    python src/train_pretrain.py --max-steps 2500
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GPTConfig  # noqa: E402
from dataset import get_batch, load_memmap  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer_bpe import load_tokenizer  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PROMPTS = [
    "从前，有一个小女孩",
    "一天，小明和小红一起去公园",
    "小猫看见了一只小鸟",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--run-name", type=str, default="pretrain")
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=40)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pos-emb", type=str, default="learned", choices=["learned", "sinusoidal", "none"]
    )
    parser.add_argument("--no-tie", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def lr_at_step(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * coeff)


@torch.no_grad()
def evaluate(model: GPT, data, args: argparse.Namespace, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(data, args.batch_size, args.block_size, device)
        _, loss, _, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample_texts(model: GPT, tokenizer, eos_id: int, max_new_tokens: int = 60) -> list[dict]:
    samples = []
    for prompt in SAMPLE_PROMPTS:
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(
            x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50, eos_id=eos_id
        )
        text = tokenizer.decode(out[0].tolist())
        samples.append({"prompt": prompt, "text": text})
    return samples


def save_checkpoint(path: Path, model: GPT, optimizer, step: int, config: GPTConfig, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
            "train_args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_steps = min(args.max_steps, 20)
        args.eval_interval = 10
        args.eval_iters = 3
        args.batch_size = min(args.batch_size, 4)
        args.log_interval = 1
        args.warmup_steps = min(args.warmup_steps, 5)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cpu"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints" / args.run_name
    log_dir = out_dir / "logs"
    sample_dir = out_dir / "samples"
    for path in (ckpt_dir, log_dir, sample_dir):
        path.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        tie_weights=not args.no_tie,
        pos_emb=args.pos_emb,
    )
    model = GPT(config)
    print(f"参数量: {model.num_parameters()/1e6:.3f}M (非嵌入 {model.num_parameters(True)/1e6:.3f}M)")
    print(f"配置: {config}")

    train_data = load_memmap(data_dir / "train.bin")
    val_data = load_memmap(data_dir / "val.bin")
    print(f"训练 token: {len(train_data):,}  验证 token: {len(val_data):,}")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.lr
    )
    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"从 {args.resume} 恢复，step={start_step}")

    log_path = log_dir / f"{args.run_name}_log.csv"
    log_exists = log_path.exists() and start_step > 0
    log_file = open(log_path, "a" if log_exists else "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not log_exists:
        writer.writerow(
            ["step", "tokens_seen", "lr", "train_loss", "val_loss", "ppl", "elapsed_sec", "tokens_per_sec"]
        )

    samples_path = sample_dir / f"{args.run_name}_samples.json"
    all_samples: list[dict] = []
    if samples_path.exists() and start_step > 0:
        all_samples = json.loads(samples_path.read_text(encoding="utf-8"))

    model.train()
    t0 = time.time()
    running_loss = None
    tokens_seen = start_step * args.batch_size * args.block_size
    frac_steps = {int(args.max_steps * f) for f in (0.25, 0.5, 0.75, 1.0)}

    print(f"开始训练：{args.max_steps} 步，batch={args.batch_size}，block={args.block_size}")
    try:
        for step in range(start_step, args.max_steps):
            lr = lr_at_step(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, args.batch_size, args.block_size, device)
            _, loss, _, _ = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            tokens_seen += x.numel()
            loss_value = loss.item()
            running_loss = loss_value if running_loss is None else 0.95 * running_loss + 0.05 * loss_value

            if (step + 1) % args.log_interval == 0 or step == start_step:
                elapsed = time.time() - t0
                tps = (step + 1 - start_step) * args.batch_size * args.block_size / max(1e-9, elapsed)
                print(
                    f"step {step+1:5d}/{args.max_steps} | loss {running_loss:.4f} | "
                    f"lr {lr:.2e} | {tps:6.0f} tok/s | {elapsed/60:5.1f} min",
                    flush=True,
                )
                writer.writerow(
                    [
                        step + 1,
                        tokens_seen,
                        f"{lr:.8f}",
                        f"{running_loss:.6f}",
                        "",
                        "",
                        f"{elapsed:.1f}",
                        f"{tps:.1f}",
                    ]
                )
                log_file.flush()

            if (step + 1) % args.eval_interval == 0 or step + 1 == args.max_steps:
                val_loss = evaluate(model, val_data, args, device)
                ppl = math.exp(min(20.0, val_loss))
                elapsed = time.time() - t0
                tps = (step + 1 - start_step) * args.batch_size * args.block_size / max(1e-9, elapsed)
                print(f"  [eval] step {step+1} val_loss {val_loss:.4f} ppl {ppl:.2f}", flush=True)
                writer.writerow(
                    [
                        step + 1,
                        tokens_seen,
                        f"{lr:.8f}",
                        "",
                        f"{val_loss:.6f}",
                        f"{ppl:.4f}",
                        f"{elapsed:.1f}",
                        f"{tps:.1f}",
                    ]
                )
                log_file.flush()

                samples = sample_texts(model, tokenizer, eos_id)
                model.train()  # generate() 内部会切到 eval，采样后切回训练模式
                all_samples.append({"step": step + 1, "val_loss": val_loss, "samples": samples})
                samples_path.write_text(
                    json.dumps(all_samples, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(ckpt_dir / "ckpt_best.pt", model, optimizer, step + 1, config, args)

            if step + 1 in frac_steps:
                save_checkpoint(
                    ckpt_dir / f"ckpt_step{step+1}.pt", model, optimizer, step + 1, config, args
                )
            if (step + 1) % 500 == 0:
                save_checkpoint(ckpt_dir / "ckpt_last.pt", model, optimizer, step + 1, config, args)

    except KeyboardInterrupt:
        print("收到中断，保存当前状态...")
        save_checkpoint(ckpt_dir / "ckpt_last.pt", model, optimizer, step + 1, config, args)
    finally:
        log_file.close()

    # 最终保存
    save_checkpoint(ckpt_dir / "ckpt_last.pt", model, optimizer, args.max_steps, config, args)
    elapsed = time.time() - t0
    meta = {
        "run_name": args.run_name,
        "config": config.to_dict(),
        "args": vars(args),
        "params_m": model.num_parameters() / 1e6,
        "params_non_embedding_m": model.num_parameters(True) / 1e6,
        "train_tokens_available": int(len(train_data)),
        "val_tokens_available": int(len(val_data)),
        "tokens_seen": int(tokens_seen),
        "epochs_equivalent": tokens_seen / max(1, len(train_data)),
        "best_val_loss": best_val,
        "best_ppl": math.exp(min(20.0, best_val)),
        "elapsed_sec": elapsed,
        "avg_tokens_per_sec": tokens_seen / max(1e-9, elapsed),
    }
    (log_dir / f"{args.run_name}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
