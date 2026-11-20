"""数据集与批处理工具。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

EOS = "<|endoftext|>"


def build_prompt(instruction: str, input_text: str = "") -> str:
    """构造用户侧 prompt（不包含回答）。"""
    user = instruction if not input_text else f"{instruction}\n{input_text}"
    return f"<|user|>{user}<|assistant|>"


def build_response(response: str) -> str:
    """构造回答侧文本，末尾加结束符。"""
    return f"{response}{EOS}"


class PretrainDataset(Dataset):
    """把 token id 序列按 block_size 切窗。"""

    def __init__(self, bin_path: str | Path, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.num_samples = max(0, len(self.data) - block_size - 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        chunk = self.data[index : index + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def load_memmap(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
):
    """随机采样一个 batch；返回 (x, y)，y 是 x 右移一位。"""
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    if device != "cpu":
        x, y = x.to(device), y.to(device)
    return x, y


class SFTDataset(Dataset):
    """监督微调数据集：prompt 部分的 label 置为 -100，只训练回答部分。"""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt_ids = tokenizer.encode(item["prompt"]).ids
                response_ids = tokenizer.encode(item["response"]).ids
                if not response_ids:
                    continue
                input_ids = (prompt_ids + response_ids)[:max_length]
                # 因果语言模型的标签要右移一位：最后一个 prompt token 负责预测
                # 第一个回答 token，因此 prompt 只 mask 到倒数第二个位置。
                prompt_mask = [-100] * max(0, len(prompt_ids) - 1)
                labels = (prompt_mask + response_ids + [-100])[:max_length]
                if all(label == -100 for label in labels):
                    continue
                self.items.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                        "prompt": item["prompt"],
                        "response": item["response"],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        return (
            torch.tensor(item["input_ids"], dtype=torch.long),
            torch.tensor(item["labels"], dtype=torch.long),
        )


def make_collate_fn(pad_id: int):
    def collate(batch):
        max_len = max(len(x) for x, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            input_ids[i, : len(x)] = x
            labels[i, : len(y)] = y
        return input_ids, labels

    return collate


def read_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
