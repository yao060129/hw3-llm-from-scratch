# results 目录说明

| 子目录/文件 | 内容 |
|---|---|
| `checkpoints/pretrain/` | 预训练检查点：`ckpt_best.pt`、`ckpt_step550/1100/1650/2200.pt`、`ckpt_last.pt` |
| `checkpoints/sft_sentiment_full/` | 情感分类全参数微调检查点 |
| `checkpoints/sft_sentiment_lora/` | 情感分类 LoRA 检查点（含 `lora_only.pt`） |
| `checkpoints/sft_instruct_full/` | 指令微调检查点 |
| `logs/pretrain_log.csv` | 预训练逐步日志（loss、lr、PPL） |
| `logs/pretrain_meta.json` | 预训练汇总（参数量、token 数、耗时、最佳 PPL） |
| `logs/sft_*_log.csv` | 微调逐步日志 |
| `logs/sft_*_test_metrics.json` | 情感分类测试指标与混淆矩阵 |
| `logs/eval_*_ppl.json` | 独立困惑度评估 |
| `logs/ablations.json` | 消融实验结果 |
| `logs/tests.json` | 因果性、KV Cache、分词 round-trip、过拟合测试 |
| `samples/pretrain_samples.json` | 各检查点的生成样例 |
| `samples/sft_*` | 微调后的情感预测与指令样例 |
| `figures/` | 报告配图（PNG） |
| `tables.md` / `summary.json` | 自动汇总的 Markdown 表格与 JSON 摘要 |

检查点文件较大（单个约 20～60 MB），默认不提交到 Git；可按 README 中的命令重新训练生成。
