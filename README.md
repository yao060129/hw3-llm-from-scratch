# 作业三：从零构建大语言模型

GitHub 仓库：https://github.com/yao060129/hw3-llm-from-scratch

本目录包含作业三的全部代码、数据、实验结果与报告。项目在**无 GPU 的普通 PC** 上完成：从零实现 GPT-2 风格的小型语言模型（约 5.0M 参数），训练字节级 BPE 分词器，在中文 TinyStories 上预训练，并完成情感分类微调与指令微调。

## 目录结构

```
作业三/
├── src/
│   ├── config.py            # 模型与训练配置
│   ├── tokenizer_bpe.py     # 字节级 BPE 训练/加载/统计
│   ├── bpe_demo.py          # 从零实现的最小 BPE（教学用）
│   ├── prepare_data.py      # 下载数据的清洗、BPE、token 打包、SFT 数据构造
│   ├── dataset.py           # 预训练 / SFT 数据集与批处理
│   ├── model.py             # GPT 架构、因果注意力、KV Cache、生成
│   ├── lora.py              # 从零实现 LoRA
│   ├── train_pretrain.py    # 预训练主程序
│   ├── train_sft.py         # 情感分类 / 指令微调（全参 + LoRA）
│   ├── evaluate.py          # PPL、情感分类、指令样例、自由生成
│   ├── run_ablations.py     # 位置编码/权重共享/层数消融
│   ├── run_tests.py         # 因果性、KV Cache、分词 round-trip、单 batch 过拟合
│   ├── make_figures.py      # 生成报告图表
│   ├── make_tables.py       # 汇总实验表格
│   └── build_report.py      # Markdown -> Word -> PDF
├── data/                    # 原始数据与处理后的 .bin（大文件见下方下载说明）
├── results/
│   ├── checkpoints/         # 预训练 / SFT 权重
│   ├── logs/                # 训练日志、指标 JSON
│   ├── figures/             # 报告配图
│   └── samples/             # 生成样例
├── report/
│   ├── 作业三_实验报告.md
│   ├── 作业三_实验报告.docx
│   └── 2453-孟珂瑶-1120243260-作业三.pdf
├── requirements.txt
└── README.md
```

## 环境

- Python 3.12
- PyTorch 2.6.0（CPU 版即可；有 CUDA 也可直接运行）
- 其他依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 复现步骤

在 `作业三` 目录下执行：

```bash
# 1. 数据准备（首次运行会读取 data/TinyStories_all_data_zh_1M.tar.gz）
python src/prepare_data.py --max-stories 30000

# 2. 正确性测试（可选）
python src/run_tests.py

# 3. 预训练（标准档：2200 步，约 43 分钟 / 8 线程 CPU）
python src/train_pretrain.py --max-steps 2200 --batch-size 16 --block-size 256 \
    --warmup-steps 150 --eval-interval 200 --run-name pretrain

# 4. 情感分类微调（全参数）
python src/train_sft.py --task sentiment --method full --epochs 1

# 5. 情感分类微调（LoRA）
python src/train_sft.py --task sentiment --method lora --epochs 1

# 6. 指令微调
python src/train_sft.py --task instruct --method full --epochs 1

# 7. 评估
python src/evaluate.py --ckpt results/checkpoints/sft_sentiment_full/ckpt_best.pt \
    --task sentiment --split test --name sft_sentiment_full
python src/evaluate.py --ckpt results/checkpoints/sft_sentiment_lora/ckpt_best.pt \
    --task sentiment --split test --name sft_sentiment_lora

# 8. 消融实验（可选，约 10 分钟）
python src/run_ablations.py

# 9. 图表与表格
python src/make_figures.py
python src/make_tables.py

# 10. 生成报告 PDF（需要本机安装 Microsoft Word）
python src/build_report.py
```

> `build_report.py` 会优先复用 `../作业二/tools/pypandoc` 中的 pandoc；如果该目录不存在，
> 请先执行 `pip install pypandoc_binary`，或修改脚本中的 `PYPANDOC_DIR`。

## 如何查看模型效果

### 1. 看报告与图表

- 最终报告：`report/2453-孟珂瑶-1120243260-作业三.pdf`
- 训练曲线：`results/figures/fig04_pretrain.png`
- 情感分类混淆矩阵：`results/figures/fig08_confusion.png`
- 汇总表格：`results/tables.md`
- 训练日志：`results/logs/pretrain_log.csv`、`results/logs/sft_*_log.csv`
- 不同阶段的生成样例：`results/samples/pretrain_samples.json`

### 2. 直接运行模型（命令行 demo）

```bash
# 故事续写（预训练模型）
python src/demo.py story --prompt "从前，有一个小女孩" --max-new-tokens 80

# 情感分类（全参数微调模型）
python src/demo.py sentiment --text "这家酒店的服务非常好，下次还会再来。"
python src/demo.py sentiment --text "房间太小，隔音很差，非常失望。"

# 指令微调模型
python src/demo.py instruct --instruction "向上级汇报任务已完成"
python src/demo.py instruct --instruction "把下面这句话翻译成英文" --input "你好，很高兴认识你。"
```

`src/demo.py` 会自动加载对应的检查点（`results/checkpoints/.../ckpt_best.pt`），不需要手动指定；也可以用 `--ckpt` 换成任意检查点（例如 LoRA 检查点）。

### 3. 查看检查点

检查点是 PyTorch 的 `state_dict`，包含 `model`、`config`、`step` 等字段：

```python
import torch
ckpt = torch.load("results/checkpoints/pretrain/ckpt_best.pt", map_location="cpu")
print(ckpt["config"], ckpt["step"])
```

更推荐直接用 `src/demo.py` 或 `src/evaluate.py`，它们会处理配置、分词器和生成逻辑。

## 数据来源

| 数据 | 用途 | 地址 |
|---|---|---|
| 中文 TinyStories | 预训练 | https://hf-mirror.com/datasets/52AI/TinyStoriesZh |
| Alpaca-zh | 指令微调 | https://hf-mirror.com/datasets/shibing624/alpaca-zh |
| ChnSentiCorp | 情感分类微调 | https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp |

`data/` 中的原始压缩包与 `results/checkpoints/` 中的权重文件体积较大，不放入 Git 仓库；下载地址与生成命令见上表与复现步骤。`data/*.bin` 与 `data/tokenizer.json` 可由 `prepare_data.py` 自动重新生成。

## 主要结果

| 实验 | 配置 | 结果 |
|---|---|---|
| 预训练 | 5.023M 参数，2200 步，901 万 token，2.5 epoch | 验证损失 4.620，困惑度 101.5，CPU 约 42.6 分钟 |
| 情感分类（零样本） | 预训练模型直接生成 | 准确率 49.3%（1200 条全部为无效格式） |
| 情感分类（全参数微调） | 1 epoch，lr=2e-4 | 准确率 85.75%，F1 86.27% |
| 情感分类（LoRA r=8） | 147,456 可训练参数（2.85%） | 准确率 72.08%，F1 75.95% |
| 指令微调 | 6000 条 Alpaca-zh，1 epoch | 学会部分回答格式，复杂指令仍不稳定 |

数据与完整日志见 `results/`，报告见 `report/2453-孟珂瑶-1120243260-作业三.pdf`。

## 参考资源

- rasbt/LLMs-from-scratch：https://github.com/rasbt/LLMs-from-scratch
- 配套视频：https://space.bilibili.com/3546869640726821
- nanoGPT：https://github.com/karpathy/nanoGPT
- 作业提供的博客：https://blog.csdn.net/Y525698136/article/details/150011090 ；https://zhuanlan.zhihu.com/p/1986604312797934203

代码为本人独立实现，参考了上述开源项目的结构与思路，并在报告中注明了出处。
