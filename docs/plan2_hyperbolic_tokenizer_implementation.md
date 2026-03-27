# OpenOneRec 方案二（双曲 Itemic Token 空间）工程实施文档

> 范围：只讨论方案二，即在 OpenOneRec 的 tokenizer / semantic ID 构建阶段，将当前欧式 RQ-Kmeans 改为双曲空间版本。
>
> 目标：先做一个工程上可执行、可对照、可在 one8/HPC4 上逐步推进的实现方案，再决定是否进入完整 pretrain。

## 1. 先给结论

### 1.1 方案二的真正改造点不在 `pretrain/`，而在 `tokenizer/ + pid2sid + data`

当前仓库的 Itemic Token 不是在线由模型生成的，而是先通过 tokenizer 离线把 item embedding 量化成 3 层代码，再在数据构造阶段把 `pid -> sid` 映射写进训练样本。

也就是说，方案二的第一步不是直接跑 pretrain，而是先改通下面这条链路：

```text
embedding parquet
-> Hyperbolic RQ-Kmeans
-> pid2sid parquet
-> 重新生成 pretrain / SFT 数据
-> 再进入 Stage1 / Stage2 pretrain
```

### 1.2 方案二不能直接复用官方 `OneRec-1.7B-pretrain` 当训练起点

这一点和方案一不同。

方案一只改输出头，所以可以从官方 `OneRec-1.7B-pretrain` 分叉；但方案二改的是 **Itemic Token 的定义本身**。一旦 `pid -> [s_a, s_b, s_c]` 变了：

- `video_ad_pid2sid.parquet` / `product_pid2sid.parquet` 会变；
- 所有含 SID 的 pretrain / SFT 数据都会变；
- itemic token 的语义也变了。

因此，**官方 `OneRec-1.7B-pretrain` 只能作为参考，不是方案二的正式起点**。  
方案二要做严格比较，正确对照应当是：

```text
B2: 欧式 tokenizer -> 同样的 Stage1/Stage2 pretrain -> 同样的 SFT/benchmark
E2: 双曲 tokenizer -> 同样的 Stage1/Stage2 pretrain -> 同样的 SFT/benchmark
```

### 1.3 当前仓库里并不存在“必须在 embedding lookup 前做 Log Map”这一步

这点要特别纠正。

你给出的原始想法里提到：

> Transformer 的 embedding lookup 仍在欧式空间，需要在 embedding 层前增加双曲 -> 欧式的 Log Map

但在 **当前 OpenOneRec 仓库实现** 里，tokenizer 只是离线产生离散 token ID，对模型侧暴露的是：

- `<s_a_i>`
- `<s_b_i>`
- `<s_c_i>`

模型训练时看到的是普通离散 token，embedding lookup 仍然是标准词表嵌入，不直接消费双曲坐标。  
所以：

- **仅做方案二的 tokenizer 版本时，不需要额外在 Transformer 前加 Log Map；**
- 只有当你进一步想把双曲码本几何显式注入 embedding 初始化或 regularization 时，才需要考虑 `exp/log map` 桥接。

这意味着：**方案二第一轮可以只改 semantic ID 构建，不必同时改 Qwen 骨干。**

---

## 2. 当前仓库里和方案二直接相关的事实

### 2.1 当前 tokenizer 是欧式 residual K-means

当前实现集中在 [res_kmeans.py](/Applications/workspace/OpenOneRec/tokenizer/res_kmeans.py)：

- 训练使用 `faiss.Kmeans(... spherical=False)`；
- assignment 使用平方欧式距离；
- centroid update 使用标准 K-means 中心；
- 每一层做 residual quantization。

这说明当前 OpenOneRec 的 tokenizer 完全是欧式版本，没有任何双曲几何。

### 2.2 tokenizer 输出是 `pid + codes`，下游数据构造吃的是 `pid + sid`

当前推理脚本 [infer_res_kmeans.py](/Applications/workspace/OpenOneRec/tokenizer/infer_res_kmeans.py) 输出：

- `pid`
- `codes`

但所有数据脚本实际读取的是：

- `pid`
- `sid`

比如 [item_understand.py](/Applications/workspace/OpenOneRec/data/onerec_data/pretrain/item_understand.py)、[video_rec.py](/Applications/workspace/OpenOneRec/data/onerec_data/pretrain/video_rec.py)、[video_rec.py](/Applications/workspace/OpenOneRec/data/onerec_data/sft/video_rec.py) 都会：

- `pd.read_parquet(pid2sid_path)`
- `dict(zip(df['pid'], df['sid']))`

因此，方案二除了训练双曲 tokenizer 之外，还必须补一层桥接：

```text
codes.parquet -> pid2sid.parquet
```

否则根本接不到现有数据流水线。

### 2.3 数据生成脚本把 `sid` 直接拼成 `<s_a_x><s_b_y><s_c_z>`

当前 data 脚本中 SID 的格式是固定的，例如：

```text
<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>
```

见 [item_understand.py](/Applications/workspace/OpenOneRec/data/onerec_data/pretrain/item_understand.py)、[video_rec.py](/Applications/workspace/OpenOneRec/data/onerec_data/sft/video_rec.py) 等。

这意味着方案二第一轮最稳的做法是：

- 仍然保留 3 层代码；
- 仍然保留 `<s_a_i>/<s_b_i>/<s_c_i>` 的 token 命名；
- 仍然让 `sid` 存 `[c0, c1, c2]` 这样的 triplet。

这样可以最大化复用现有数据与 pretrain 管线。

### 2.4 当前 itemic 词表扩展默认假设每层 8192 个 token

词表扩展逻辑在 [expand_qwen3_vocab.py](/Applications/workspace/OpenOneRec/pretrain/tools/model_converter/expand_qwen3_vocab.py)。

这里有两个强约束：

- token 顺序必须是 `s_a_0..N-1`, `s_b_0..N-1`, `s_c_0..N-1`, `<|sid_begin|>`, `<|sid_end|>`；
- 当前仓库默认每层 `8192`。

所以方案二第一轮为了减少变量，建议：

- **先保持 `n_layers=3` 与 `codebook_size=8192` 不变；**
- 只把欧式量化改成双曲量化；
- 不要第一轮就同时改码本大小。

“双曲空间表达效率更高，所以码本可缩小”这件事可以作为后续 ablation，但不应该和第一轮主实验绑在一起。

---

## 3. 方案二的正确实验定义

### 3.1 第一轮不要直接和官方 final / pretrain checkpoint 比

因为方案二改的是 semantic ID 构造方式，一旦 SID 变了，官方 checkpoint 中 itemic token 的语义绑定就不再一致。

因此第一轮实验应当定义为：

```text
B2_euclidean_tokenizer:
Qwen3 base + 欧式 tokenizer + 重建 pid2sid + 重建 pretrain 数据 + Stage1/Stage2 pretrain

E2_hyperbolic_tokenizer:
Qwen3 base + 双曲 tokenizer + 重建 pid2sid + 重建 pretrain 数据 + Stage1/Stage2 pretrain
```

然后两边再走同样的 post-train / benchmark。

### 3.2 第一轮推荐先做 tokenizer 质量验证，不要直接开全量 pretrain

最短路径应该是：

1. 先在 embedding parquet 上做 Euclidean vs Hyperbolic RQ-Kmeans 对照；
2. 验证码本是否稳定、重构误差是否合理、长尾 item 是否更分散；
3. 生成一版新的 `pid2sid`；
4. 用新的 `pid2sid` 构出小规模 pretrain 样本 sanity check；
5. 确认 token 分布、数据格式和 vocab 全对上后，再跑 Stage1/Stage2。

方案二本质上是“上游表征工程”，不应该一上来就消耗 8 卡去赌 tokenizer 是否可用。

---

## 4. 代码层面的推荐改造方式

### 4.1 不要直接把 `res_kmeans.py` 改烂，建议并行保留欧式与双曲两套实现

推荐新增：

- `tokenizer/hyperbolic_ops.py`
- `tokenizer/hyper_res_kmeans.py`
- `tokenizer/build_pid2sid.py`

保留原有：

- [res_kmeans.py](/Applications/workspace/OpenOneRec/tokenizer/res_kmeans.py)
- [train_res_kmeans.py](/Applications/workspace/OpenOneRec/tokenizer/train_res_kmeans.py)
- [infer_res_kmeans.py](/Applications/workspace/OpenOneRec/tokenizer/infer_res_kmeans.py)

更稳的做法是让 `train_res_kmeans.py` / `infer_res_kmeans.py` 加一个开关，例如：

```text
--space euclidean|hyperbolic
```

这样可以：

- 让 B2 和 E2 用同一套 CLI；
- 减少重复脚本；
- 保持输出格式一致。

### 4.2 双曲版本第一轮建议只实现 Poincare Ball 必要算子

第一轮不需要追求把所有几何细节做得很花。

够用的最小集合：

- `expmap0`
- `logmap0`
- Möbius addition
- Poincare distance
- Fréchet mean 迭代

建议自己在 repo 内实现最小算子，而不是一开始强依赖 `geoopt`。原因是：

- 当前仓库没有这类依赖；
- one8 / HPC4 环境受限，少一个外部库少一个变量；
- 方案二第一轮核心是验证离散码本构建，不是做通用双曲学习框架。

### 4.3 Wrapped Normal 初始化可以先做“切空间高斯 + expmap0”

第一版不必追求复杂采样器。工程上更实用的是：

1. 在原点切空间采样高斯向量；
2. 控制范数；
3. 通过 `expmap0` 投影到球内；
4. 作为初始 codebook。

这足以作为 Wrapped Normal 的工程近似版本，用于第一轮 POC。

### 4.4 assignment 与 centroid update 的推荐实现

建议分层 residual 流程保持和当前实现一致，只替换几何：

1. 当前 residual 向量先做 `expmap0` 进入球；
2. 用双曲距离做 assignment；
3. 每个 cluster 用 Fréchet mean 更新中心；
4. residual 更新时，不建议直接做“球上减法”的复杂版本作为第一步；
5. 第一轮可以采用更稳的折中做法：
   - 在 tangent space 表示 residual；
   - codebook center 也通过 `logmap0` 拉回切空间做 residual subtraction；
   - assignment 仍基于球内距离。

也就是说，第一轮可以做一个 **hybrid hyperbolic RQ**：

- cluster assignment 在双曲球上；
- residual accumulation 在原点切空间。

这样更容易稳定，也更适合先验证方向。

---

## 5. 方案二真正会影响哪些下游模块

### 5.1 `pid2sid` 映射文件必须重建

当前下游默认使用：

- `raw_data/onerec_data/video_ad_pid2sid.parquet`
- `raw_data/onerec_data/product_pid2sid.parquet`

这些必须由新的 tokenizer 重新生成。  
否则你虽然换了 tokenizer，data pipeline 仍然在吃旧 SID。

### 5.2 pretrain / SFT 数据都必须重建

所有依赖 `pid2sid` 的任务都要重新导出，包括：

- pretrain 的 `video_rec` / `item_understand`
- SFT 的 `video_rec` / `interactive_rec` / `label_cond_rec` / `label_pred` / `ad_rec` / `product_rec` / `item_understand`

也就是说，方案二不只是“重训 tokenizer”这么简单，而是会影响：

```text
tokenizer
-> pid2sid
-> RecIF task parquet
-> split_data_pretrain / split_data_sft
-> pretrain / post-train
```

### 5.3 如果保持 3x8192，不需要改数据 SID 模板

只要保持：

- `n_layers = 3`
- `codebook_size = 8192`

那么：

- `<s_a_i>/<s_b_i>/<s_c_i>` 模板不需要改；
- `SID_FORMAT` 不需要改；
- `itemic_id_range` 也更容易沿用现有逻辑。

这也是为什么第一轮不建议缩码本。

---

## 6. 方案二的正确执行顺序

### Step 0. 固化一条 Euclidean tokenizer baseline

先用当前仓库 tokenizer 做一条干净 baseline：

- 同一份 embedding 数据；
- 同一份 `n_layers=3, codebook_size=8192, dim=4096`；
- 记录：
  - 重构 MSE
  - relative loss
  - dead code 比例
  - 每层码字使用分布

### Step 1. 实现 hyperbolic tokenizer POC

目标不是一开始训练超大规模 tokenizer，而是先在小样本上验证：

- 双曲 assignment 正常
- Fréchet mean 收敛
- 编码输出仍然是 3 维整数 code
- reconstruction / usage 统计可计算

### Step 2. 生成新的 `pid2sid`

建议新增一个桥接脚本，例如：

```text
tokenizer/build_pid2sid.py
```

输入：

- `pid`
- `codes`

输出：

- `pid`
- `sid`

其中 `sid` 保持为 `[c0, c1, c2]` list，兼容现有 data 脚本。

### Step 3. 重新生成 RecIF 数据

用新的 `pid2sid` 覆盖数据入口，再跑：

- [run.sh](/Applications/workspace/OpenOneRec/data/onerec_data/run.sh)
- [prepare_pretrain.sh](/Applications/workspace/OpenOneRec/data/prepare_pretrain.sh)
- [prepare_sft.sh](/Applications/workspace/OpenOneRec/data/prepare_sft.sh)

### Step 4. 扩 itemic vocab

如果仍用 `3 x 8192`，则直接沿用现有词表扩展逻辑。  
如果你改了码本大小，那么必须同步修改：

- `expand_qwen3_vocab.py` 的 `vocab_size_per_layer`
- 所有 itemic token range 设定
- 对应模型配置与测试样例

### Step 5. 再跑 Stage1 / Stage2 pretrain

方案二的正式 pretrain 不应从 `OneRec-1.7B-pretrain` 出发，而应从：

- `Qwen3-1.7B` base + expanded vocab

然后重新做：

1. Itemic-Text Alignment
2. Full-Parameter Co-Pretraining

### Step 6. 再决定是否接 SFT / benchmark

只有在 tokenizer 质量和 pretrain 稳定性都过关后，再接 SFT/benchmark。  
否则很难判断问题出在：

- 双曲 tokenizer；
- pretrain recipe；
- 还是 post-train。

---

## 7. one8 / HPC4 上这轮该怎么跑

### 7.1 tokenizer POC 先用 CPU，别一开始占满 8 卡

这是方案二和当前 pretrain 线的最大区别。

Fréchet mean 迭代和双曲 assignment 在第一版实现里更适合先：

- CPU 小样本；
- 或单卡小批量；
- 先验证数值与速度。

one8 的 8 卡更应该留给：

- 重新生成 itemic vocab 后的 Stage1/Stage2 pretrain；
- 而不是最初级的聚类调试。

### 7.2 第一轮不要改码本大小

在 one8 / HPC4 上的第一轮方案二，建议参数固定为：

- `n_layers = 3`
- `codebook_size = 8192`
- `dim = 4096`

只改“欧式 -> 双曲”的几何，不再叠加额外变量。

### 7.3 pretrain 启动前的必要验收

在真正启动 8 卡 pretrain 前，至少保证下面几件事：

1. `pid2sid` parquet 能被现有 data 脚本直接读取；
2. 生成出的 pretrain / SFT parquet 中，SID 文本格式正确；
3. 新的 itemic vocab 与 `sid` triplet 取值范围一致；
4. tokenizer 没有大面积 dead code；
5. Hyperbolic vs Euclidean 的输出格式完全一致。

---

## 8. 第一轮应该看什么指标

### 8.1 tokenizer 阶段

第一轮至少记录：

- reconstruction MSE
- relative loss
- 每层 code usage 分布
- dead code ratio
- 高频 / 长尾 item 的码字覆盖情况
- 每层 prefix 的聚类均衡性

如果双曲版本在长尾 item 上更均匀、dead code 更少、或层次分离更好，才说明它值得进入 pretrain。

### 8.2 pretrain 阶段

进入 pretrain 后再看：

- Stage1 / Stage2 loss 是否稳定下降
- itemic token loss 与 text token loss 是否正常
- 是否出现 NaN / 发散
- checkpoint 可否正常保存和恢复

### 8.3 不要第一轮就用 final benchmark 判生死

方案二首先是 tokenizer / semantic ID 的上游改造。  
第一轮最核心的问题是：

**“双曲量化是否产生了更好的层次化 item code，并且没有把训练数据链路搞坏？”**

---

## 9. 这轮最容易踩的坑

### 坑 1：直接拿官方 `OneRec-1.7B-pretrain` 继续训

这会导致“新 SID 与旧 item embedding 语义不一致”，实验结论不可信。

### 坑 2：只改 `tokenizer/`，不重建 `pid2sid`

这样下游数据仍然吃旧 SID，等于没做方案二。

### 坑 3：第一轮就缩小码本

这会把“几何变化”和“码本容量变化”混在一起，无法归因。

### 坑 4：把双曲坐标直接硬塞给 Transformer

当前仓库不是这种接口设计。  
第一轮应当保持 tokenizer 只输出离散 codes。

### 坑 5：一开始就上 8 卡全量 pretrain

如果 tokenizer 输出格式或 `pid2sid` 桥接有问题，会直接浪费大量卡时。

---

## 10. 给 Codex 的具体任务拆分

### Task A：文档与接口

1. 新增方案二实施文档
2. 设计 `euclidean|hyperbolic` 统一 CLI
3. 设计 `codes -> sid` 桥接脚本接口

### Task B：tokenizer POC

1. 实现最小双曲算子
2. 实现 Hyperbolic RQ-Kmeans
3. 在小样本 embedding parquet 上跑通 train / infer
4. 产出 `pid + codes`

### Task C：数据链路

1. 把 `codes` 转成 `pid2sid`
2. 重新生成 RecIF pretrain / SFT parquet
3. 验证 SID 文本格式与 token range

### Task D：pretrain

1. 从 Qwen3 base + expanded itemic vocab 起跑
2. 重跑 Stage1 / Stage2
3. 记录与欧式 tokenizer baseline 的差异

---

## 11. 最终一句话执行指令

**方案二不是“直接基于官方 pretrain 改一点再跑”，而是“先把欧式 RQ-Kmeans 换成双曲版本，重新生成 `pid2sid` 和含 SID 的训练数据，再从 Qwen3 base + expanded vocab 重新跑 Stage1/Stage2 pretrain”；第一轮保持 3 层、每层 8192 不变，只改几何，不改码本容量。**
