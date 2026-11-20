"""生成报告所需的全部图表。"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GPTConfig  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer_bpe import load_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
DATA = ROOT / "data"
FIG.mkdir(parents=True, exist_ok=True)

# 参考 dataviz 调色板：分类色按固定顺序使用
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#6250d6"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "blue_red",
    ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3b0a8", "#e34948", "#a32120"],
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["SimHei", "Microsoft YaHei"],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _box(ax, x, y, w, h, title, sub, color=BLUE, fill="#eaf2fd"):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2, edgecolor=color, facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=10, color=INK, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
            fontsize=8, color=MUTED)


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    stages = [
        ("① 数据准备", "TinyStories-zh\n清洗/去重/划分", "#eaf2fd", BLUE),
        ("② 分词", "字节级 BPE\n词表 12000", "#eaf2fd", BLUE),
        ("③ 词嵌入 + 位置编码", "Token Embedding\nLearned Pos Embedding", "#eaf2fd", BLUE),
        ("④ GPT 架构", "6 层 Transformer\n5.0M 参数", "#eaf2fd", BLUE),
        ("⑤ 预训练", "下一词预测\n900 万 token", "#eaf2fd", BLUE),
        ("⑥ 微调", "情感分类 / 指令\n全参 + LoRA", "#eaf2fd", BLUE),
        ("⑦ 推理评估", "PPL / 生成 / 分类\nKV Cache", "#eaf2fd", BLUE),
    ]
    positions = [(0.15, 2.55), (2.55, 2.55), (4.95, 2.55), (7.35, 2.55),
                 (7.35, 0.75), (4.95, 0.75), (2.55, 0.75)]
    w, h = 1.95, 1.2
    for (x, y), (title, sub, fill, color) in zip(positions, stages):
        _box(ax, x, y, w, h, title, sub, color=color, fill=fill)
    # 箭头
    arrows = [
        ((2.10, 3.15), (2.55, 3.15)),
        ((4.50, 3.15), (4.95, 3.15)),
        ((6.90, 3.15), (7.35, 3.15)),
        ((8.325, 2.55), (8.325, 1.95)),
        ((7.35, 1.35), (6.90, 1.35)),
        ((4.95, 1.35), (4.50, 1.35)),
        ((2.55, 1.35), (2.10, 1.35)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=12, color=MUTED, linewidth=1.0))
    ax.text(5.0, 3.92, "从零构建大模型的完整流程", ha="center", fontsize=13,
            color=INK, fontweight="bold")
    fig.savefig(FIG / "fig01_pipeline.png")
    plt.close(fig)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(3.35, 8.0)
    ax.axis("off")

    def block(y, text, color=BLUE, height=0.55, fill="#eaf2fd", fontsize=9.5):
        box = FancyBboxPatch((1.55, y), 4.1, height,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             linewidth=1.2, edgecolor=color, facecolor=fill)
        ax.add_patch(box)
        ax.text(3.6, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)

    def arrow(y1, y2):
        ax.add_patch(FancyArrowPatch((3.6, y1), (3.6, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=MUTED, linewidth=1.0))

    ax.text(3.6, 7.88, "GPT（decoder-only）", ha="center", fontsize=13,
            color=INK, fontweight="bold")
    block(7.12, "输出层  Linear(V, d)  +  Softmax", color=VIOLET, fill="#f0edfb")
    block(6.30, "最终 LayerNorm", color=VIOLET, fill="#f0edfb")

    # Transformer Block：标题在顶部，内部结构在下方
    box = FancyBboxPatch((1.55, 5.05), 4.1, 1.15,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=1.2, edgecolor=BLUE, facecolor="#eaf2fd")
    ax.add_patch(box)
    ax.text(3.6, 5.95, "Transformer Block × 6", ha="center", va="center",
            fontsize=9.5, color=INK, fontweight="bold")
    ax.text(3.6, 5.62, "LN → 多头因果自注意力 → 残差", ha="center", va="center",
            fontsize=8, color=MUTED)
    ax.text(3.6, 5.35, "LN → FFN(4d) → 残差", ha="center", va="center",
            fontsize=8, color=MUTED)

    block(4.10, "位置编码  +  Token Embedding（共享权重）", color=AQUA, fill="#e8f7f1")
    block(3.42, "输入 token 序列  $x_1 \\ldots x_T$", color=MUTED, fill="#f4f4f2", height=0.5)
    for y1, y2 in [(3.92, 4.10), (4.65, 5.05), (6.20, 6.30), (6.85, 7.12)]:
        arrow(y1, y2)

    ax.text(5.9, 5.65, "d = 192\nh = 6\nFFN = 768\nL = 6", fontsize=8.5,
            color=MUTED, va="top")
    fig.savefig(FIG / "fig02_architecture.png")
    plt.close(fig)


def fig_tokenizer() -> None:
    tokenizer = load_tokenizer(DATA / "tokenizer.json")
    stats = json.loads((DATA / "data_stats.json").read_text(encoding="utf-8"))
    lengths = []
    with open(DATA / "tinystories_zh.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            text = json.loads(line)["text"]
            lengths.append(len(tokenizer.encode(text).ids))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    ax.hist(lengths, bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axvline(np.percentile(lengths, 50), color=ORANGE, linewidth=1.6, label="中位数")
    ax.axvline(np.percentile(lengths, 90), color=VIOLET, linewidth=1.6, label="90% 分位")
    ax.set_xlabel("每条故事的 token 数")
    ax.set_ylabel("故事数量")
    ax.set_title("(a) 故事长度分布（2000 条样本）", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(axis="y")

    ax = axes[1]
    labels = ["字符级\n(1 字/token)", f"本项目 BPE\n({stats['chars_per_token']:.2f} 字/token)"]
    values = [1.0, stats["chars_per_token"]]
    bars = ax.bar(labels, values, color=[MUTED, BLUE], width=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05,
                f"{value:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylabel("每个 token 平均字符数")
    ax.set_title("(b) 分词压缩率", fontsize=11)
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_tokenizer.png")
    plt.close(fig)


def fig_pretrain() -> None:
    log_path = RESULTS / "logs" / "pretrain_log.csv"
    df = pd.read_csv(log_path)
    train = df[df["train_loss"].notna()].copy()
    val = df[df["val_loss"].notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    ax.plot(train["step"], train["train_loss"], color=BLUE, linewidth=1.6, label="训练损失")
    ax.plot(val["step"], val["val_loss"], color=ORANGE, linewidth=1.6,
            marker="o", markersize=4, label="验证损失")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("交叉熵损失")
    ax.set_title("(a) 损失曲线", fontsize=11)
    ax.legend()
    ax.grid(axis="y")

    ax = axes[1]
    ax.plot(val["step"], val["ppl"], color=AQUA, linewidth=1.8, marker="o", markersize=4)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("验证困惑度 PPL")
    ax.set_title("(b) 困惑度", fontsize=11)
    ax.grid(axis="y")

    ax = axes[2]
    ax.plot(train["step"], train["lr"], color=VIOLET, linewidth=1.8)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("学习率")
    ax.set_title("(c) warmup + cosine 学习率", fontsize=11)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig04_pretrain.png")
    plt.close(fig)


def _load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def fig_attention() -> None:
    ckpt = RESULTS / "checkpoints" / "pretrain" / "ckpt_best.pt"
    if not ckpt.exists():
        print("跳过注意力图：找不到", ckpt)
        return
    model, config = _load_model(ckpt)
    tokenizer = load_tokenizer(DATA / "tokenizer.json")
    text = "从前，有一个小女孩住在一个小村庄里，她每天都会去森林里玩。"
    ids = tokenizer.encode(text).ids[:16]
    # 字节级 BPE 的单个 token 可能只是多字节汉字的一部分，直接 decode([id]) 会得到乱码；
    # 因此用“前缀解码的增量”来还原每个 token 对应的可读文本。
    tokens = []
    prev = ""
    for i in range(1, len(ids) + 1):
        cur = tokenizer.decode(ids[:i])
        if cur.startswith(prev):
            piece = cur[len(prev) :]
        else:
            piece = cur[-2:]
        prev = cur
        piece = piece.replace("\n", "⏎").replace(" ", "·")
        piece = "".join(ch if ch.isprintable() else "·" for ch in piece)
        tokens.append(piece if piece else "·")
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        _, _, _, attns = model(x, return_attn=True)

    selected = [(0, 0), (2, 2), (5, 5)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (layer, head) in zip(axes, selected):
        att = attns[layer][0, head].numpy()
        im = ax.imshow(att, cmap=SEQ_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"第 {layer+1} 层 · 第 {head+1} 头", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="注意力权重")
    fig.suptitle("训练后模型的注意力分布（16 个 token）", fontsize=12)
    fig.savefig(FIG / "fig05_attention.png")
    plt.close(fig)


def fig_posemb() -> None:
    ckpt = RESULTS / "checkpoints" / "pretrain" / "ckpt_best.pt"
    if not ckpt.exists():
        print("跳过位置编码图：找不到", ckpt)
        return
    model, config = _load_model(ckpt)
    if config.pos_emb != "learned":
        return
    w = model.wpe.weight.detach()
    w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = (w @ w.T).numpy()

    # 左图：前 64 个位置的相关性（隐藏对角线，放大 0~0.3 的差异）
    n = 64
    sub = sim[:n, :n].copy()
    np.fill_diagonal(sub, np.nan)
    cmap = SEQ_CMAP.copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    im = ax.imshow(sub, cmap=cmap, vmin=0.0, vmax=0.3, origin="lower")
    ax.set_xlabel("位置 $j$")
    ax.set_ylabel("位置 $i$")
    ax.set_title("(a) 前 64 个位置的余弦相似度（对角线已隐藏）", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="余弦相似度")

    # 右图：平均相似度随位置距离的衰减
    distances = np.arange(1, 129)
    means = np.array(
        [np.mean([sim[i, i + d] for i in range(256 - d)]) for d in distances]
    )
    ax = axes[1]
    ax.plot(distances, means, color=BLUE, linewidth=1.8)
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    ax.set_xlabel("位置距离 $d=|i-j|$")
    ax.set_ylabel("平均余弦相似度")
    ax.set_title("(b) 相似度随距离的衰减", fontsize=10)
    ax.grid(axis="y")

    fig.tight_layout()
    fig.savefig(FIG / "fig06_posemb.png")
    plt.close(fig)


def fig_sft_loss() -> None:
    paths = {
        "全参数微调": RESULTS / "logs" / "sft_sentiment_full_log.csv",
        "LoRA": RESULTS / "logs" / "sft_sentiment_lora_log.csv",
    }
    if not any(path.exists() for path in paths.values()):
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = {"全参数微调": BLUE, "LoRA": ORANGE}
    for label, path in paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        train = df[df["train_loss"].notna()]
        val = df[df["val_loss"].notna()]
        ax.plot(train["step"], train["train_loss"], color=colors[label],
                linewidth=1.4, label=f"{label} 训练")
        if len(val):
            ax.plot(val["step"], val["val_loss"], color=colors[label],
                    linewidth=1.2, linestyle="none", marker="o", markersize=4)
    ax.set_xlabel("微调步数")
    ax.set_ylabel("损失（仅回答 token）")
    ax.set_title("情感分类微调损失", fontsize=11)
    ax.legend()
    ax.grid(axis="y")
    fig.savefig(FIG / "fig07_sft_loss.png")
    plt.close(fig)


def _confusion_panel(ax, cm, title):
    cm = np.array(cm)
    im = ax.imshow(cm, cmap=SEQ_CMAP)
    labels = ["负面", "正面"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            color = "white" if value > cm.max() * 0.55 else INK
            ax.text(j, i, f"{value}\n{value/total*100:.1f}%", ha="center", va="center",
                    color=color, fontsize=10)


def fig_confusion() -> None:
    paths = {
        "全参数微调": RESULTS / "logs" / "sft_sentiment_full_test_metrics.json",
        "LoRA": RESULTS / "logs" / "sft_sentiment_lora_test_metrics.json",
    }
    available = {k: v for k, v in paths.items() if v.exists()}
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.4))
    if len(available) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, available.items()):
        data = json.loads(path.read_text(encoding="utf-8"))
        _confusion_panel(ax, data["confusion_matrix"], label)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_confusion.png")
    plt.close(fig)


def fig_lora() -> None:
    paths = {
        "全参数微调": RESULTS / "logs" / "sft_sentiment_full_test_metrics.json",
        "LoRA": RESULTS / "logs" / "sft_sentiment_lora_test_metrics.json",
    }
    data = {k: json.loads(v.read_text(encoding="utf-8")) for k, v in paths.items() if v.exists()}
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    labels = list(data.keys())
    colors = [BLUE, ORANGE][: len(labels)]

    ax = axes[0]
    values = [data[k]["trainable_params"] / 1e6 for k in labels]
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02,
                f"{value:.2f}M", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("可训练参数量（百万）")
    ax.set_title("(a) 可训练参数", fontsize=11)
    ax.grid(axis="y")

    ax = axes[1]
    values = [data[k]["accuracy"] * 100 for k in labels]
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("测试准确率（%）")
    ax.set_ylim(0, 100)
    ax.set_title("(b) 情感分类准确率", fontsize=11)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig09_lora.png")
    plt.close(fig)


def fig_ablation() -> None:
    path = RESULTS / "logs" / "ablations.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data = sorted(data, key=lambda x: x["best_val_loss"])
    labels = [x["label"] for x in data]
    values = [x["best_val_loss"] for x in data]
    colors = [BLUE if "基线" in label else "#b9c8dd" for label in labels]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    for bar, value in zip(bars, values):
        ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("验证损失（300 步短程训练）")
    ax.set_title("消融实验：不同配置的验证损失（越低越好）", fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_ablation.png")
    plt.close(fig)


def main() -> None:
    setup_style()
    fig_pipeline()
    fig_architecture()
    fig_tokenizer()
    fig_pretrain()
    fig_attention()
    fig_posemb()
    fig_sft_loss()
    fig_confusion()
    fig_lora()
    fig_ablation()
    print("图表已生成到", FIG)


if __name__ == "__main__":
    main()
