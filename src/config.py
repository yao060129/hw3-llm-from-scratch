"""模型与训练配置。

所有实验共享这里的默认配置；命令行可以覆盖其中的字段。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    """GPT-2 风格 decoder-only 模型的配置。"""

    vocab_size: int = 12000      # 词表大小（由 BPE 分词器决定）
    block_size: int = 256        # 上下文长度
    n_layer: int = 6             # Transformer Block 层数
    n_head: int = 6              # 多头注意力的头数
    n_embd: int = 192            # 隐藏维度
    dropout: float = 0.1         # dropout 概率
    bias: bool = True            # 线性层是否使用偏置（GPT-2 风格）
    tie_weights: bool = True     # 词嵌入与输出头是否共享权重
    pos_emb: str = "learned"     # learned / sinusoidal / none

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """预训练默认超参数（标准档，约 1000 万 token）。"""

    batch_size: int = 16
    block_size: int = 256
    max_steps: int = 2500
    warmup_steps: int = 200
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 40
    log_interval: int = 10
    seed: int = 42
