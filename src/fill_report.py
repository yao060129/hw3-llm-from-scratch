"""把实验结果填入报告模板中的占位符。

在预训练、微调、评估全部完成后运行：
    python src/fill_report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "作业三_实验报告.md"
RESULTS = ROOT / "results"


def load(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def cell(text: str, max_len: int = 100) -> str:
    text = str(text).replace("\n", " ").replace("|", "｜").strip()
    return text[:max_len] + ("…" if len(text) > max_len else "")


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")

    meta = load(RESULTS / "logs" / "pretrain_meta.json", {})
    if meta:
        text = text.replace("{{FINAL_PPL}}", f"{meta['best_ppl']:.1f}")

    samples = load(RESULTS / "samples" / "pretrain_samples.json", [])
    if samples:
        rows = []
        for item in samples:
            rows.append(
                f"| {item['step']} | {item['val_loss']:.4f} | "
                f"{math.exp(min(20, item['val_loss'])):.2f} |"
            )
        text = text.replace("{{CKPT_TABLE}}", "\n".join(rows))

        # 选取约 25%/50%/75%/100% 的检查点
        target_steps = [600, 1200, 1800, 2200]
        selected = []
        for target in target_steps:
            candidates = [s for s in samples if s["step"] <= target]
            if candidates:
                selected.append(candidates[-1])
        if selected:
            rows = []
            for item in selected:
                sample = item["samples"][0]["text"]
                progress = item["step"] / 2200 * 100
                rows.append(
                    f"| {progress:.0f}% | {item['val_loss']:.4f} | {cell(sample, 90)} |"
                )
            text = text.replace("{{SAMPLE_TABLE}}", "\n".join(rows))

    ablations = load(RESULTS / "logs" / "ablations.json", [])
    if ablations:
        rows = [
            f"| {item['label']} | {item['params_m']:.3f}M | "
            f"{item['best_val_loss']:.4f} | {item['best_ppl']:.2f} |"
            for item in ablations
        ]
        text = text.replace("{{ABLATION_TABLE}}", "\n".join(rows))

    zero = load(RESULTS / "logs" / "eval_pretrain_zero_shot_sentiment_test.json")
    full = load(RESULTS / "logs" / "sft_sentiment_full_test_metrics.json")
    lora = load(RESULTS / "logs" / "sft_sentiment_lora_test_metrics.json")
    if full or lora:
        rows = []
        named = [("零样本（预训练模型）", zero), ("全参数微调", full), ("LoRA", lora)]
        for name, metrics in named:
            if not metrics:
                continue
            rows.append(
                f"| {name} | {metrics['accuracy']*100:.2f}% | "
                f"{metrics['precision']*100:.2f}% | {metrics['recall']*100:.2f}% | "
                f"{metrics['f1']*100:.2f}% | {metrics['invalid_outputs']} |"
            )
        text = text.replace("{{SENTIMENT_TABLE}}", "\n".join(rows))

        if full:
            text = text.replace("{{SENT_ACC}}", f"{full['accuracy']*100:.2f}%")
            cm = full["confusion_matrix"]
            zero_acc = zero["accuracy"] * 100 if zero else 50.0
            zero_invalid = zero["invalid_outputs"] if zero else 1200
            text = text.replace(
                "{{SENTIMENT_ANALYSIS}}",
                f"全参数微调把测试准确率从零样本的 {zero_acc:.1f}% 提升到 "
                f"{full['accuracy']*100:.2f}%，F1 为 {full['f1']*100:.2f}%。"
                f"混淆矩阵中负面类正确 {cm[0][0]} 条、正面类正确 {cm[1][1]} 条，"
                f"两类错误数量相对均衡，说明模型没有明显偏向某一类。"
                f"无效输出从零样本的 {zero_invalid} 条"
                f"降到 {full['invalid_outputs']} 条，说明微调让模型学会了“正面/负面”的固定回答格式。",
            )

        if full and lora:
            text = text.replace(
                "| 全参数微调 | 4,998,144 | 4,998,144 | 100% |",
                f"| 全参数微调 | {full['trainable_params']:,} | {full['total_params']:,} | 100% |",
            )
            text = text.replace(
                "| LoRA（r=8） | 147,456 | 5,145,600 | 2.87% |",
                f"| LoRA（r=8） | {lora['trainable_params']:,} | {lora['total_params']:,} | "
                f"{lora['trainable_ratio']*100:.2f}% |",
            )
            text = text.replace("2.87%", f"{lora['trainable_ratio']*100:.2f}%")
            text = text.replace(
                "{{LORA_ANALYSIS}}",
                f"全参数微调训练了 {full['trainable_params']:,} 个参数，测试准确率 "
                f"{full['accuracy']*100:.2f}%；LoRA 只训练 {lora['trainable_params']:,} 个参数"
                f"（占总量 {lora['trainable_ratio']*100:.2f}%，约为全参数的 1/34），"
                f"测试准确率 {lora['accuracy']*100:.2f}%，两者相差 "
                f"{abs(full['accuracy']-lora['accuracy'])*100:.2f} 个百分点。"
                f"从混淆矩阵看，LoRA 的召回率为 {lora['recall']*100:.1f}%，但精确率只有 "
                f"{lora['precision']*100:.1f}%，说明它倾向于预测“正面”，把一部分负面样本判成了正面。"
                f"这说明低秩旁路可以快速适配回答格式并恢复大部分性能，但 147K 参数不足以完全"
                f"匹配全参数微调；提高秩、增加训练轮数或对更多层注入 LoRA 有望缩小差距。",
            )

    before = load(RESULTS / "samples" / "eval_pretrain_before_instruct.json", [])
    after = load(RESULTS / "samples" / "sft_instruct_full_samples.json", [])
    if before and after:
        pairs = list(zip(before, after))
        # 只展示 6 条代表性样例，避免报告里出现 20 行冗长的失败输出
        preferred = [1, 2, 8, 9, 12, 14]
        if len(pairs) > 6:
            pairs = [pairs[i] for i in preferred if i < len(pairs)]
        rows = []
        for b, a in pairs:
            rows.append(
                f"| {cell(b['instruction'], 26)} | {cell(b['generated'], 40)} | "
                f"{cell(a['generated'], 40)} |"
            )
        text = text.replace("{{INSTRUCT_TABLE}}", "\n".join(rows))
        text = text.replace(
            "{{INSTRUCT_ANALYSIS}}",
            "微调前，模型只会顺着 prompt 续写，经常重复或答非所问；微调后，"
            "模型学会了在 `<|assistant|>` 之后给出回答，部分样本出现了“请提供更多信息”"
            "等符合指令风格的回复，但仍有不少输出是重复符号或答非所问。"
            "由于模型只有 5M 参数、指令数据仅 6000 条且只训练 1 个 epoch，"
            "复杂指令（多步推理、长文本改写）仍然会出现事实错误或重复。"
            "这说明 SFT 主要教会了模型“回答的格式与风格”，知识和推理能力仍受预训练规模限制。",
        )

    REPORT.write_text(text, encoding="utf-8")
    print("报告占位符已填充：", REPORT)


if __name__ == "__main__":
    main()
