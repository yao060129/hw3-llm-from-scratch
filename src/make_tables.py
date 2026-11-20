"""汇总实验数据，生成 Markdown 表格，便于写入报告。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def _load(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    lines: list[str] = []
    summary: dict = {}

    stats = _load(DATA / "data_stats.json")
    if stats:
        summary["data"] = stats
        lines += [
            "## 数据与分词",
            "",
            "| 项目 | 数值 |",
            "|---|---:|",
            f"| 训练故事数 | {stats['num_stories_train']:,} |",
            f"| 验证故事数 | {stats['num_stories_val']:,} |",
            f"| 训练字符数 | {stats['train_chars']:,} |",
            f"| 训练 token 数 | {stats['train_tokens']:,} |",
            f"| 验证 token 数 | {stats['val_tokens']:,} |",
            f"| 每个 token 字符数 | {stats['chars_per_token']:.3f} |",
            f"| 词表大小 | {stats['vocab_size']:,} |",
            "",
        ]

    meta = _load(RESULTS / "logs" / "pretrain_meta.json")
    if meta:
        summary["pretrain"] = meta
        lines += [
            "## 预训练",
            "",
            "| 项目 | 数值 |",
            "|---|---:|",
            f"| 参数量 | {meta['params_m']:.3f} M |",
            f"| 训练 token | {meta['tokens_seen']:,} |",
            f"| 等效 epoch | {meta['epochs_equivalent']:.2f} |",
            f"| 最佳验证损失 | {meta['best_val_loss']:.4f} |",
            f"| 最佳困惑度 | {meta['best_ppl']:.2f} |",
            f"| 训练耗时 | {meta['elapsed_sec']/60:.1f} 分钟 |",
            f"| 吞吐 | {meta['avg_tokens_per_sec']:.0f} token/s |",
            "",
        ]

    samples = _load(RESULTS / "samples" / "pretrain_samples.json")
    if samples:
        summary["checkpoints"] = [
            {"step": s["step"], "val_loss": s["val_loss"]} for s in samples
        ]
        lines += ["## 预训练检查点", "", "| 步数 | 验证损失 | 验证 PPL |", "|---:|---:|---:|"]
        for s in samples:
            lines.append(
                f"| {s['step']} | {s['val_loss']:.4f} | {__import__('math').exp(s['val_loss']):.2f} |"
            )
        lines.append("")

    for method in ("full", "lora"):
        metrics = _load(RESULTS / "logs" / f"sft_sentiment_{method}_test_metrics.json")
        if metrics:
            summary[f"sft_sentiment_{method}"] = metrics
            lines += [
                f"## 情感分类微调（{method}）",
                "",
                "| 指标 | 数值 |",
                "|---|---:|",
                f"| 可训练参数 | {metrics['trainable_params']:,} |",
                f"| 总参数 | {metrics['total_params']:,} |",
                f"| 可训练比例 | {metrics['trainable_ratio']*100:.2f}% |",
                f"| 准确率 | {metrics['accuracy']*100:.2f}% |",
                f"| 精确率 | {metrics['precision']*100:.2f}% |",
                f"| 召回率 | {metrics['recall']*100:.2f}% |",
                f"| F1 | {metrics['f1']*100:.2f}% |",
                f"| 无效输出 | {metrics['invalid_outputs']} |",
                "",
            ]

    ablations = _load(RESULTS / "logs" / "ablations.json")
    if ablations:
        summary["ablations"] = ablations
        lines += [
            "## 消融实验",
            "",
            "| 配置 | 参数量 | 验证损失 | 验证 PPL |",
            "|---|---:|---:|---:|",
        ]
        for item in ablations:
            lines.append(
                f"| {item['label']} | {item['params_m']:.3f}M | "
                f"{item['best_val_loss']:.4f} | {item['best_ppl']:.2f} |"
            )
        lines.append("")

    (RESULTS / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
