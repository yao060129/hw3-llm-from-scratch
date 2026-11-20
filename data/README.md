# data 目录说明

本目录保存原始数据与 `prepare_data.py` 生成的处理结果。原始压缩包、`.bin` 与 SFT JSONL 体积较大，默认不提交到 Git（见 `.gitignore`）。

## 需要手动下载的原始文件

在 `作业三/data/` 下执行：

```bash
# 中文 TinyStories（约 260 MB）
curl -L -o TinyStories_all_data_zh_1M.tar.gz \
  https://hf-mirror.com/datasets/52AI/TinyStoriesZh/resolve/main/TinyStories_all_data_zh_1M.tar.gz

# Alpaca-zh 指令数据（约 35 MB）
curl -L -o alpaca_gpt4_data_zh.json \
  https://hf-mirror.com/datasets/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json

# ChnSentiCorp 三个 parquet 分片
curl -L -o senti_train.parquet \
  https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp/resolve/main/data/train-00000-of-00001-02f200ca5f2a7868.parquet
curl -L -o senti_validation.parquet \
  https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp/resolve/main/data/validation-00000-of-00001-405befbaa3bcf1a2.parquet
curl -L -o senti_test.parquet \
  https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp/resolve/main/data/test-00000-of-00001-5372924f059fe767.parquet
```

## 由脚本生成的文件

运行 `python src/prepare_data.py --max-stories 30000` 后会生成：

| 文件 | 说明 |
|---|---|
| `tinystories_zh.jsonl` | 清洗去重后的 3 万篇故事 |
| `tokenizer.json` | 12000 词表的字节级 BPE |
| `train.bin` / `val.bin` | uint16 格式的 token id 序列 |
| `sft_sentiment_*.jsonl` | 情感分类微调数据（train/validation/test） |
| `sft_instruct_*.jsonl` | 指令微调数据（train/val/eval） |
| `data_stats.json` | 数据与分词统计 |
