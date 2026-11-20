"""监督微调：情感分类任务与通用指令微调，支持全参数微调和 LoRA。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GPTConfig  # noqa: E402
from dataset import SFTDataset, make_collate_fn, read_jsonl  # noqa: E402
from lora import apply_lora, count_trainable, lora_state_dict  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer_bpe import load_tokenizer  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sentiment", choices=["sentiment", "instruct"])
    parser.add_argument("--method", type=str, default="full", choices=["full", "lora"])
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(ROOT / "results/checkpoints/pretrain/ckpt_best.pt"),
    )
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-subset", type=int, default=300)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_model(ckpt_path: str, method: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    # 先加载基座权重，再注入 LoRA（否则 state_dict 的键名对不上）
    model.load_state_dict(ckpt["model"])
    if method == "lora":
        apply_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    return model, config, ckpt


def make_loader(
    path: Path,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    max_samples: int | None = None,
):
    dataset = SFTDataset(path, tokenizer, max_length=max_length)
    if max_samples is not None:
        dataset.items = dataset.items[:max_samples]
    pad_id = tokenizer.token_to_id("<|pad|>")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(pad_id),
        num_workers=0,
        drop_last=False,
    )
    return dataset, loader


@torch.no_grad()
def evaluate_loss(model, loader, max_batches: int | None = None) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        _, loss, _, _ = model(x, targets=y)
        total += loss.item() * x.size(0)
        count += x.size(0)
    model.train()
    return total / max(1, count)


def parse_label(text: str) -> int:
    head = text.strip()[:12]
    if "正面" in head and "负面" not in head:
        return 1
    if "负面" in head and "正面" not in head:
        return 0
    if "正面" in head:
        return 1
    if "负面" in head:
        return 0
    return -1


@torch.no_grad()
def predict_sentiment(model, tokenizer, items, eos_id: int, max_new_tokens: int = 6):
    model.eval()
    preds, labels, outputs = [], [], []
    for item in items:
        prompt_ids = tokenizer.encode(item["prompt"]).ids
        prompt_ids = prompt_ids[: model.config.block_size - max_new_tokens]
        x = torch.tensor([prompt_ids], dtype=torch.long)
        out = model.generate(
            x, max_new_tokens=max_new_tokens, temperature=0.0, eos_id=eos_id, use_cache=True
        )
        generated = tokenizer.decode(out[0, len(prompt_ids) :].tolist())
        pred = parse_label(generated)
        preds.append(pred)
        labels.append(int(item["label"]))
        outputs.append({"text": item.get("text", ""), "gold": int(item["label"]), "pred": pred, "generated": generated})
    return preds, labels, outputs


def compute_sentiment_metrics(preds, labels) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    valid_preds = [p if p in (0, 1) else 0 for p in preds]
    acc = accuracy_score(labels, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, valid_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, valid_preds, labels=[0, 1]).tolist()
    invalid = sum(1 for p in preds if p not in (0, 1))
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "invalid_outputs": invalid,
        "num_samples": len(labels),
    }


@torch.no_grad()
def generate_instruct_samples(model, tokenizer, items, eos_id: int, max_new_tokens: int = 48):
    model.eval()
    results = []
    for item in items:
        prompt_ids = tokenizer.encode(item["prompt"]).ids
        prompt_ids = prompt_ids[: model.config.block_size - max_new_tokens]
        x = torch.tensor([prompt_ids], dtype=torch.long)
        out = model.generate(
            x, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40, eos_id=eos_id
        )
        generated = tokenizer.decode(out[0, len(prompt_ids) :].tolist())
        results.append(
            {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "reference": item.get("output", ""),
                "generated": generated,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"sft_{args.task}_{args.method}"
    if args.epochs is None:
        args.epochs = 2 if args.task == "sentiment" else 1
    if args.lr is None:
        args.lr = 2e-4 if args.method == "full" else 1e-3
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.eval_interval = 10
        args.eval_subset = 20
        args.max_train_samples = 40
        args.max_val_items = 20
        args.max_test_items = 20
        args.log_interval = 1

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints" / args.run_name
    log_dir = out_dir / "logs"
    sample_dir = out_dir / "samples"
    for path in (ckpt_dir, log_dir, sample_dir):
        path.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    pad_id = tokenizer.token_to_id("<|pad|>")

    model, config, base_ckpt = load_model(
        args.ckpt, args.method, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    trainable, total = count_trainable(model)
    print(f"方法: {args.method} | 可训练参数 {trainable:,} / 总参数 {total:,} "
          f"({trainable/total*100:.2f}%)")
    print(f"模型配置: {config}")

    if args.task == "sentiment":
        train_path = data_dir / "sft_sentiment_train.jsonl"
        val_path = data_dir / "sft_sentiment_validation.jsonl"
        test_path = data_dir / "sft_sentiment_test.jsonl"
    else:
        train_path = data_dir / "sft_instruct_train.jsonl"
        val_path = data_dir / "sft_instruct_val.jsonl"
        test_path = data_dir / "sft_instruct_eval.jsonl"

    train_dataset, train_loader = make_loader(
        train_path,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=True,
        max_samples=args.max_train_samples,
    )
    val_dataset, val_loader = make_loader(
        val_path, tokenizer, args.max_length, args.batch_size, shuffle=False
    )
    test_items = read_jsonl(test_path)
    val_items = read_jsonl(val_path)
    if args.max_test_items:
        test_items = test_items[: args.max_test_items]
    if args.max_val_items:
        val_items = val_items[: args.max_val_items]
    print(f"训练样本 {len(train_dataset)} | 验证样本 {len(val_dataset)} | 测试样本 {len(test_items)}")

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(10, int(0.05 * total_steps))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = log_dir / f"{args.run_name}_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["step", "epoch", "lr", "train_loss", "val_loss", "val_acc", "elapsed_sec"])

    # 情感任务用准确率（越大越好），指令任务用 -val_loss（越大越好）
    best_metric = -float("inf")
    best_path = ckpt_dir / "ckpt_best.pt"
    global_step = 0
    t0 = time.time()
    model.train()

    for epoch in range(args.epochs):
        model.train()
        running_loss = None
        for x, y in train_loader:
            lr = optimizer.param_groups[0]["lr"]
            _, loss, _, _ = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            loss_value = loss.item()
            running_loss = loss_value if running_loss is None else 0.95 * running_loss + 0.05 * loss_value

            if global_step % args.log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps} "
                    f"loss {running_loss:.4f} lr {lr:.2e} {elapsed/60:.1f}min",
                    flush=True,
                )
                writer.writerow([global_step, epoch + 1, f"{lr:.8f}", f"{running_loss:.6f}", "", "", f"{elapsed:.1f}"])
                log_file.flush()

            if global_step % args.eval_interval == 0:
                val_loss = evaluate_loss(model, val_loader, max_batches=30)
                val_acc = ""
                if args.task == "sentiment":
                    subset = val_items[: args.eval_subset]
                    preds, labels, _ = predict_sentiment(model, tokenizer, subset, eos_id)
                    metrics = compute_sentiment_metrics(preds, labels)
                    val_acc = metrics["accuracy"]
                    print(f"  [eval] val_loss {val_loss:.4f} val_acc {val_acc:.4f}", flush=True)
                    model.train()
                    metric = val_acc
                else:
                    print(f"  [eval] val_loss {val_loss:.4f}", flush=True)
                    metric = -val_loss

                writer.writerow([global_step, epoch + 1, f"{lr:.8f}", "", f"{val_loss:.6f}",
                                 f"{val_acc}" if val_acc != "" else "", f"{time.time()-t0:.1f}"])
                log_file.flush()

                if metric > best_metric:
                    best_metric = metric
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": config.to_dict(),
                            "method": args.method,
                            "task": args.task,
                            "base_ckpt": args.ckpt,
                            "step": global_step,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                        },
                        best_path,
                    )
                    if args.method == "lora":
                        torch.save(lora_state_dict(model), ckpt_dir / "lora_only.pt")

        # 每个 epoch 结束做一次完整验证
        val_loss = evaluate_loss(model, val_loader)
        if args.task == "sentiment":
            preds, labels, outputs = predict_sentiment(model, tokenizer, val_items, eos_id)
            metrics = compute_sentiment_metrics(preds, labels)
            print(f"[epoch {epoch+1}] val_loss {val_loss:.4f} val_acc {metrics['accuracy']:.4f}", flush=True)
            if metrics["accuracy"] > best_metric:
                best_metric = metrics["accuracy"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config.to_dict(),
                        "method": args.method,
                        "task": args.task,
                        "base_ckpt": args.ckpt,
                        "step": global_step,
                        "val_loss": val_loss,
                        "val_acc": metrics["accuracy"],
                    },
                    best_path,
                )
                if args.method == "lora":
                    torch.save(lora_state_dict(model), ckpt_dir / "lora_only.pt")
        else:
            print(f"[epoch {epoch+1}] val_loss {val_loss:.4f}", flush=True)
            metric = -val_loss
            if metric > best_metric:
                best_metric = metric
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config.to_dict(),
                        "method": args.method,
                        "task": args.task,
                        "base_ckpt": args.ckpt,
                        "step": global_step,
                        "val_loss": val_loss,
                        "val_acc": "",
                    },
                    best_path,
                )
                if args.method == "lora":
                    torch.save(lora_state_dict(model), ckpt_dir / "lora_only.pt")

    log_file.close()
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "method": args.method,
            "task": args.task,
            "base_ckpt": args.ckpt,
            "step": global_step,
        },
        ckpt_dir / "ckpt_last.pt",
    )

    # 用最佳模型做最终测试
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model"])

    if args.task == "sentiment":
        preds, labels, outputs = predict_sentiment(model, tokenizer, test_items, eos_id)
        metrics = compute_sentiment_metrics(preds, labels)
        metrics.update(
            {
                "method": args.method,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_ratio": trainable / total,
                "epochs": args.epochs,
                "train_samples": len(train_dataset),
            }
        )
        (log_dir / f"{args.run_name}_test_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(sample_dir / f"{args.run_name}_test_predictions.jsonl", "w", encoding="utf-8") as f:
            for row in outputs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("测试指标:", json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        eval_items = test_items[:20]
        results = generate_instruct_samples(model, tokenizer, eval_items, eos_id)
        (sample_dir / f"{args.run_name}_samples.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已生成 {len(results)} 条指令微调样例")

    meta = {
        "run_name": args.run_name,
        "task": args.task,
        "method": args.method,
        "config": config.to_dict(),
        "args": vars(args),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
        "total_steps": global_step,
        "elapsed_sec": time.time() - t0,
    }
    (log_dir / f"{args.run_name}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
