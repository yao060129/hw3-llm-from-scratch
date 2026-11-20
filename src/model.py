"""小型 GPT 模型的完整实现。

结构对应 GPT-2（Radford et al., 2019）：
    词嵌入 + 位置编码 -> N 个 Pre-LN Transformer Block -> LayerNorm -> 线性输出头

本文件刻意手写多头因果自注意力，而不是直接调用 nn.MultiheadAttention，
以便把 Q/K/V 投影、缩放点积、因果掩码和输出投影都显式展示出来。
"""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """多头因果自注意力（带可选的 KV Cache）。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # 一次线性变换得到 Q、K、V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 下三角矩阵，1 表示允许 attend，0 表示屏蔽
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        T_total = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if past_kv is None:
            mask = self.causal_mask[:, :, :T, :T]
        else:
            past_len = T_total - T
            # 第 i 个新 token 可以看到 past_len + i 之前（含）的所有 key
            mask = torch.tril(
                torch.ones(T, T_total, device=x.device), diagonal=past_len
            ).view(1, 1, T, T_total)
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        present = (k, v) if use_cache else None
        return y, present, (att if return_attn else None)


class MLP(nn.Module):
    """Transformer 中的前馈网络：Linear -> GELU -> Linear。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-LN 的 Transformer Block。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False, return_attn=False):
        attn_out, present, att = self.attn(
            self.ln_1(x), past_kv=past_kv, use_cache=use_cache, return_attn=return_attn
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present, att


class GPT(nn.Module):
    """GPT-2 风格的语言模型。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_emb == "learned":
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_emb == "sinusoidal":
            self.register_buffer("pe", self._build_sinusoidal(config), persistent=False)
            self.wpe = None
        elif config.pos_emb == "none":
            self.wpe = None
        else:
            raise ValueError(f"未知的位置编码类型: {config.pos_emb}")

        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 论文对残差投影使用更小的初始化
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _build_sinusoidal(config: GPTConfig) -> torch.Tensor:
        pe = torch.zeros(config.block_size, config.n_embd)
        position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.n_embd, 2, dtype=torch.float)
            * (-math.log(10000.0) / config.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ):
        B, T = idx.shape
        past_len = 0 if past_kv is None else past_kv[0][0].size(2)
        if T + past_len > self.config.block_size:
            raise ValueError(
                f"序列长度 {T + past_len} 超过 block_size {self.config.block_size}"
            )

        tok_emb = self.wte(idx)
        pos_emb = 0.0
        if self.config.pos_emb == "learned":
            pos = torch.arange(past_len, past_len + T, device=idx.device)
            pos_emb = self.wpe(pos)
        elif self.config.pos_emb == "sinusoidal":
            pos_emb = self.pe[:, past_len : past_len + T]

        x = self.drop(tok_emb + pos_emb)

        presents = [] if use_cache else None
        attns = [] if return_attn else None
        for i, block in enumerate(self.h):
            block_past = None if past_kv is None else past_kv[i]
            x, present, att = block(
                x,
                past_kv=block_past,
                use_cache=use_cache,
                return_attn=return_attn,
            )
            if use_cache:
                presents.append(present)
            if return_attn:
                attns.append(att)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, presents, attns

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = (cumulative - probs) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                1, sorted_indices, sorted_logits
            )
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
        eos_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """自回归生成。use_cache=True 时使用 KV Cache。"""
        self.eval()
        # 如果生成总长度会超过上下文，退化为每次重算
        if idx.size(1) + max_new_tokens > self.config.block_size:
            use_cache = False

        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _, _, _ = self(idx_cond)
                next_token = self._sample(
                    logits[:, -1, :], temperature, top_k, top_p
                )
                idx = torch.cat([idx, next_token], dim=1)
                if eos_id is not None and bool((next_token == eos_id).all()):
                    break
            return idx

        past = None
        for step in range(max_new_tokens):
            if step == 0:
                prompt = idx[:, -self.config.block_size :]
                logits, _, past, _ = self(prompt, use_cache=True)
            else:
                logits, _, past, _ = self(
                    idx[:, -1:], past_kv=past, use_cache=True
                )
            next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, next_token], dim=1)
            if eos_id is not None and bool((next_token == eos_id).all()):
                break
        return idx

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
        return n

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for _, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay.append(param)
            else:
                no_decay.append(param)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


@torch.no_grad()
def check_causality(config: GPTConfig | None = None, device: str = "cpu") -> bool:
    """验证因果掩码：改变最后一个 token 不应影响前面位置的 logits。"""
    config = config or GPTConfig(vocab_size=128, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config).to(device).eval()
    x = torch.randint(0, config.vocab_size, (2, 12), device=device)
    logits1, _, _, _ = model(x)
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % config.vocab_size
    logits2, _, _, _ = model(x2)
    past_unchanged = torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5)
    last_changed = not torch.allclose(logits1[:, -1], logits2[:, -1], atol=1e-5)
    return bool(past_unchanged and last_changed)


@torch.no_grad()
def check_kv_cache(config: GPTConfig | None = None) -> bool:
    """验证 KV Cache 与非 Cache 的贪心生成结果一致。"""
    config = config or GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    out_cache = model.generate(prompt.clone(), 20, temperature=0.0, use_cache=True)
    out_plain = model.generate(prompt.clone(), 20, temperature=0.0, use_cache=False)
    return bool(torch.equal(out_cache, out_plain))


if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg)
    print("配置:", asdict(cfg))
    print(f"参数量: {model.num_parameters()/1e6:.3f} M")
    print(f"非嵌入参数量: {model.num_parameters(non_embedding=True)/1e6:.3f} M")
    print("因果性测试:", "通过" if check_causality() else "失败")
    print("KV Cache 测试:", "通过" if check_kv_cache() else "失败")
