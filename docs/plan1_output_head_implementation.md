# OpenOneRec 方案一（仅改输出层）实施说明

## 结论

本轮只做方案一，不重跑 tokenizer、Stage1、Stage2，也不先接 distillation / RL。

正确起点是官方公开的 pretrain checkpoint：

- `OpenOneRec/OneRec-1.7B-pretrain`
- 可选后续扩展到 `OpenOneRec/OneRec-8B-pretrain`

实验拆成三层：

1. `B0_official_final_eval`
   - 官方 final 模型仅用于 benchmark sanity check。
2. `B1_pretrain_to_sft_baseline`
   - 官方 pretrain checkpoint
   - 不改模型结构
   - 官方 SFT 流程
   - benchmark 评测
3. `E1_hyp_head_from_pretrain`
   - 同一个官方 pretrain checkpoint
   - 只改 itemic token 输出打分
   - 同样的 SFT
   - 同样的 benchmark

## 已验证事实

这些点已经和仓库现状对齐：

- `pretrain/README.md` 明确把训练链路分成 `Itemic-Text Alignment -> Full-parameter Co-Pretraining -> SFT`。
- `benchmarks/README.md` 明确 benchmark 入口是 `bash eval_script.sh <model_path> <result_name> <enable_thinking>`。
- `pretrain/examples/posttrain_sft.sh` 明确 SFT 入口独立存在。
- `pretrain/README.md` 明确 `--use_tie_weights` 用于 `0.6B / 1.7B / 4B` 小模型。

因此本轮实验不应该从 Qwen3 base model 重跑 pretrain，而应从 `OneRec-1.7B-pretrain` 分叉。

## 基线定义

本次真正 baseline 定义为：

```text
OneRec-1.7B-pretrain
-> 不改模型结构
-> posttrain_sft.sh / train_qwen3.py
-> convert_checkpoint_to_hf.sh
-> benchmark
```

注意：

- `OpenOneRec/OneRec-1.7B` 只拿来做 benchmark 环境 sanity。
- 不把官方 final 模型当本次主 baseline，因为它已经混入 SFT + distillation + RL。

## 方案一改动边界

第一轮只允许一个结构性改动：

- 只改 `itemic_id_range` 这段 token 的输出打分。

不建议在 `1.7B` 上直接替换整个 `lm_head`。更稳的实现是 hybrid logits override：

1. 正常算原始 `base_logits`
2. 只对 `itemic_id_range` 对应 slice 重新算双曲 logits
3. 覆盖回 `base_logits[..., itemic_id_range]`
4. 文本 token logits 保持原样

这样可以绕开 `1.7B` tied embeddings 的高风险改动。

## 数据和模型

本轮至少需要：

- 模型
  - `OpenOneRec/OneRec-1.7B`
  - `OpenOneRec/OneRec-1.7B-pretrain`
- 数据
  - `OpenOneRec/OpenOneRec-RecIF`
  - `OpenOneRec/OpenOneRec-General-SFT`

benchmark 默认数据目录：

```bash
export BENCHMARK_BASE_DIR="."
export BENCHMARK_DATA_DIR="../raw_data/onerec_data/benchmark_data"
export DATA_VERSION="v1.0"
```

## 执行顺序

### 1. Sanity

先用官方 final 模型做 benchmark sanity：

```bash
cd benchmarks
bash eval_script.sh /path/to/OneRec-1.7B official_1p7b false
```

调试时可以给 `eval_script.sh` 每条 python 命令加 `--sample_size 10`。

### 2. Baseline

从 `OneRec-1.7B-pretrain` 出发做一版干净 baseline：

```text
pretrain checkpoint -> SFT -> convert -> benchmark
```

要求：

- baseline 和方案一使用同一份 SFT 数据
- 同一份 dataset config
- 同一套步数 / lr / seed

### 3. 方案一

在 plan1 工作树里增加 itemic slice 的双曲重打分分支：

```text
OneRec-1.7B-pretrain
-> hybrid logits override
-> SFT
-> convert
-> benchmark
```

## 成功标准

工程成功：

1. 官方 final 模型能跑 benchmark
2. `OneRec-1.7B-pretrain` 能接 SFT
3. baseline 能完整转 HF 并评测
4. plan1 也能完整转 HF 并评测

研究成功：

- `E1` 相比 `B1` 在多个 recommendation 子任务上稳定优于 baseline
- 没明显伤害文本生成
- 训练稳定，无明显 NaN / 爆炸

## 主要风险

- 把 official final 当主 baseline，归因会脏
- 在 `1.7B` 上暴力替换整个 `lm_head`
- baseline 和 plan1 的 SFT 数据或步数不一致
- 一开始就接 RL，变量过多
- benchmark judge 配置不全导致主流程被卡住

## 当前落地策略

远端 one8 维持两套完全隔离工作树：

- `/data/user/cwu319/OpenOneRec-base`
- `/data/user/cwu319/OpenOneRec-plan1`

约束：

- `base` 只跑干净 baseline
- `plan1` 只承载方案一改动
- 两边共享 `.env`、模型、原始数据
- 两边输出目录隔离

这样可以保证后续 `B1` 和 `E1` 的结果可比，且不会发生代码互相污染。
