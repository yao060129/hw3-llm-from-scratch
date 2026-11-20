"""小规模消融实验：位置编码、权重共享、层数。

每个配置只训练 300 步（batch 8 x 128），用于相对比较，不追求最终效果。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = [
    {"name": "ablate_learned", "label": "可学习位置编码（基线）", "extra": []},
    {"name": "ablate_none", "label": "无位置编码", "extra": ["--pos-emb", "none"]},
    {"name": "ablate_sinusoidal", "label": "正弦位置编码", "extra": ["--pos-emb", "sinusoidal"]},
    {"name": "ablate_untied", "label": "不共享词嵌入/输出头", "extra": ["--no-tie"]},
    {"name": "ablate_2layer", "label": "2 层 Transformer", "extra": ["--n-layer", "2"]},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "results" / "logs" / "ablations.json"
    if out_path.exists() and not args.force:
        print(f"{out_path} 已存在，使用 --force 重新运行")
        return

    results = []
    for cfg in CONFIGS:
        meta_path = ROOT / "results" / "logs" / f"{cfg['name']}_meta.json"
        if not meta_path.exists() or args.force:
            cmd = [
                sys.executable,
                str(ROOT / "src" / "train_pretrain.py"),
                "--run-name",
                cfg["name"],
                "--max-steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
                "--block-size",
                str(args.block_size),
                "--lr",
                "6e-4",
                "--warmup-steps",
                "50",
                "--eval-interval",
                "100",
                "--eval-iters",
                "20",
                "--log-interval",
                "50",
                "--threads",
                "8",
            ] + cfg["extra"]
            print("=" * 70)
            print("运行:", cfg["label"], "->", " ".join(cmd[-4:]))
            subprocess.run(
                cmd,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONUTF8": "1"},
                check=True,
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        results.append(
            {
                "name": cfg["name"],
                "label": cfg["label"],
                "best_val_loss": meta["best_val_loss"],
                "best_ppl": meta["best_ppl"],
                "params_m": meta["params_m"],
                "elapsed_sec": meta["elapsed_sec"],
                "tokens_per_sec": meta["avg_tokens_per_sec"],
                "steps": meta["args"]["max_steps"],
                "batch_size": meta["args"]["batch_size"],
                "block_size": meta["args"]["block_size"],
                "pos_emb": meta["config"]["pos_emb"],
                "tie_weights": meta["config"]["tie_weights"],
                "n_layer": meta["config"]["n_layer"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
