"""从零实现 LoRA（Low-Rank Adaptation）。

只训练注入到线性层旁路的低秩矩阵 A、B，冻结原模型参数。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
    """把 GPT 每个 Block 中的注意力/FFN 线性层替换为 LoRALinear。

    先冻结全部基座参数（包括词嵌入、LayerNorm、输出头），
    再注入可训练的 LoRA 低秩矩阵。
    """
    for param in model.parameters():
        param.requires_grad_(False)

    for block in model.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, r, alpha, dropout)
        block.attn.c_proj = LoRALinear(block.attn.c_proj, r, alpha, dropout)
        block.mlp.c_fc = LoRALinear(block.mlp.c_fc, r, alpha, dropout)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, r, alpha, dropout)


def trainable_parameters(model) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_state_dict(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
