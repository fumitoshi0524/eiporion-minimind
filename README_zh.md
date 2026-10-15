# Eiporion-MiniMind：小型语言模型的直接 INT8 权重训练

[English report](README.md)

## 摘要

Eiporion-MiniMind 探索一个具体问题：小型 decoder-only 语言模型的投影矩阵，能否在预训练过程中始终以 INT8 格式存储，并直接接受离散权重更新？实现将 MiniMind 衍生的 Transformer 与作者开发的 Eiporion PyTorch 库结合：注意力和前馈网络使用 INT8 权重及逐行 FP32 缩放因子，词嵌入、共享输出头和归一化权重保留浮点表示。

项目包含约 63.9M 等效参数的八层模型、公开追踪的两轮训练运行，以及六项中英文零样本评测。该运行完成 39,696 个 microbatch 迭代，记录的训练损失从第 100 步的 8.0936 降到最后一步的 1.9683。对保留的本地导出检查点进行存储核算后，完整权重及缩放因子约占 **75.30 MiB**；相同唯一权重的 FP32 表示约占 **243.81 MiB**，权重存储缩减约 **3.24 倍**。多项评测接近随机选择水平，因此，本实验主要研究直接量化更新的训练行为与表示成本，不据此宣称语言理解能力具有竞争力，或量化训练与浮点训练精度相当。

## 1. 研究问题

压缩训练完成后的模型，与减少训练过程中持有的状态，是不同的问题。训练后量化从已训练的模型出发；许多量化感知训练方案在前向过程中模拟量化，同时更新浮点主权重。本项目关心的是：

> 能否把注意力和前馈投影矩阵本身保存在 INT8 张量中，并在语言模型预训练中持续更新这些整数值？

困难在于更新分辨率：优化器给出的更新可能远小于整数网格的一步，直接确定性取整会丢失这些更新。实现通过残差累积与随机取整，使较小的更新也有机会改变整数权重。

工作的具体内容包括：

- **底层实现：** 作者开发的 [Eiporion 库](https://github.com/fumitoshi0524/eiporion)，提供 INT8 线性层、自定义反向传播和量化优化器状态。
- **模型集成：** 替换每个 MiniMind 衍生 Transformer block 内的七个投影矩阵，保留浮点嵌入与输出接口。
- **实验记录：** 整理训练轨迹、六项已有评测结果，以及压缩范围明确的权重存储核算。

文中的“直接 INT8 权重训练”描述这套实现，不表示所有张量和运算都是整数，也不表示随机取整能消除全部梯度近似，更不作首次提出某类通用训练范式的优先权声明。

## 2. 模型架构与参数统计

模型沿用 [MiniMind](https://github.com/jingyaogong/minimind) 的 decoder-only 设计，使用 GQA、RoPE、RMSNorm 和 SwiGLU，并在注意力计算前对 query/key 分别进行归一化。

| 项目 | 配置 |
|---|---|
| Transformer 层数 | 8 |
| 隐藏维度 | 768 |
| Query / KV 头数 | 8 / 4 |
| 每头维度 | 96 |
| FFN 中间维度 | 2,432 |
| 词表大小 | 6,400 |
| Dropout | 0 |
| 词嵌入与输出头 | 共享浮点权重 |
| 每层量化投影 | Q、K、V、O、gate、up、down |
| 投影权重存储 | INT8 buffer，附逐行 FP32 scale |
| 配置的位置缓冲区 | 32,768 个位置 |
| 预训练序列长度 | 512 tokens |

![模型架构](img/Model_Structure.png)

*架构概览；具体精度与存储范围以上表及下文实现说明为准。*

32,768 是位置缓冲区配置，不是经过实验验证的上下文长度。代码支持可选 YaRN 缩放，但默认 `inference_rope_scaling=false`，本报告没有长上下文评测结果。

INT8 权重注册为 buffer，不会被 `sum(p.numel() for p in model.parameters())` 完整统计。也不能再向已经包含 INT8 矩阵的等效参数总数重复相加。共享的 embedding/output 权重只计一次：

| 唯一权重类别 | 元素数 |
|---|---:|
| INT8 注意力与 FFN 权重 | 58,982,400 |
| 共享 embedding/output 权重 | 4,915,200 |
| 归一化权重 | 14,592 |
| **模型等效权重总数** | **63,912,192** |
| 额外逐行缩放因子 | 63,488 |

缩放因子属于量化元数据，单独列出。模型规模因此约为 64M，而不是将量化权重重复加入总数后得到的约 123M。

不依赖检查点也可以复算：令 $h=768$、$k=384$（四个 KV 头）、$f=2432$、$L=8$、$V=6400$，投影权重数为 $L(2h^2+2hk+3hf)$，共享 embedding 权重数为 $Vh$，归一化权重数为 $L(2h+2\times96)+h$，行 scale 数为 $L(3h+2k+2f)$。

## 3. 方法

### 3.1 INT8 表示

对于形状为输出维度乘输入维度的整数矩阵 $Q$，逐行缩放后的有效权重为

$$
W_{ij}=s_i Q_{ij},\qquad Q_{ij}\in\{-127,\ldots,127\},\qquad Y=XW^\top.
$$

Transformer 中的投影层不使用 bias。初始化先生成 Kaiming-uniform 浮点矩阵，再按行量化：

$$
s_i=\frac{\max_j|W^{(0)}_{ij}|}{127},\qquad
Q_{ij}=\operatorname{clip}\left(\operatorname{round}(W^{(0)}_{ij}/s_i),-127,127\right).
$$

实现对全零行设置很小的正数下限。持久化表示是整数矩阵与行缩放因子，不再为每个量化投影额外持有一份浮点主权重。缩放因子是 buffer，并非由常规 autograd 学习的参数。

存储精度与计算精度需要区分。早期 Eiporion 实现在前向时反量化为 BF16 后进行矩阵乘法；当前版本还提供 CUDA/bitsandbytes INT8 前向路径，反向矩阵乘法仍使用 BF16。因此，不能将其描述为端到端纯整数训练或推理。

### 3.2 自定义反向传播

自定义 autograd 函数根据有效权重计算输入梯度与浮点权重梯度：

$$
G_X=G_Y W,\qquad G_W=G_Y^\top X.
$$

权重梯度累积到对应模块的缓存，再由优化器取出以更新整数 buffer。这解决了 INT8 张量无法作为普通浮点可训练参数处理的问题。

这是对离散表示采用的代理优化过程。把更新移到普通参数 autograd 之外，并不会使整数取整本身可微，也不自动证明整个梯度估计无偏。

### 3.3 优化器、残差与随机取整

`EiporionOptim` 将 AdamW 风格更新与分块量化的一阶矩、二阶矩和更新残差结合。INT8 路径的块大小为 256，使用逐块缩放和非线性压扩；剩余浮点参数由底层 bitsandbytes 优化器处理。

在固定行缩放因子下，可将更新简化为：

$$
u_t=-\eta_t\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda W_t\right),\qquad
r_t\leftarrow r_{t-1}+u_t/s,
$$

$$
\Delta Q_t=\mathcal R(r_t),\qquad
Q_{t+1}=\operatorname{clip}(Q_t+\Delta Q_t,-127,127),\qquad
r_t\leftarrow r_t-\Delta Q_t.
$$

除以 $s$ 是逐行操作。该表达式用于说明基本过程，实际实现还包含状态量化和不同版本的 scale 更新，不能忽略它们对训练轨迹的影响。

普通随机取整依据小数部分，在相邻整数间进行概率选择，未发生裁剪时满足 $\mathbb E[\mathcal R(z)\mid z]=z$。它使小于一个整数步长的更新仍能影响整数权重。

**默认优化器并不是普通无偏随机取整。** 当前预训练脚本导入的 `EiporionOptim` 在取整概率中加入动量相关偏置，默认 `sr_bias_scale=0.15`；独立的 `EiporionOptimSR` 才对应无偏取整变体。因此，不能把上述单次取整的无偏性质直接归给默认优化器或整个训练过程。本实验也没有报告二者的受控比较。

还需要注意版本差异：[早期实现](https://github.com/fumitoshi0524/eiporion/tree/bff6aab67a061b4a006b6d5a1dc6908b7c41c438) 在整数更新中保持行 scale 固定；[检查的后续版本](https://github.com/fumitoshi0524/eiporion/tree/1dde29cbd12ea2fe0b14896f4931484f787917a6) 会更新 scale，并提供更多优化器变体。公开运行记录的实际版本是 **Eiporion 0.1.0**；复现应使用该版本的实现，不能假定最新版与旧实验一致。

## 4. 训练设置

预训练使用 MiniMind 的 `pretrain_t2t_mini.jsonl`。数据加载器读取每条样本的 `text` 字段，进行分词、截断、添加首尾标记并补齐到 512 tokens；padding 对应的 label 设置为 `-100`，模型内部移位后计算下一 token 的交叉熵。

| 设置 | 运行记录与源码默认值 |
|---|---|
| Epochs | 2 |
| Microbatch | 64 条序列 |
| 梯度累积 | 4 个 microbatch |
| 单设备名义有效批次 | 256 条序列 |
| 序列长度 | 512 |
| 学习率标度 | $2\times10^{-4}$ |
| Warmup | 8,000 个 microbatch 迭代 |
| 学习率调度 | 从标度的 10% 线性预热，随后余弦衰减至 10% |
| Weight decay | 0.1 |
| Autocast 精度 | BF16 |
| 初始随机种子 | 42；每个 epoch 调整数据顺序种子 |
| 投影优化器 | `EiporionOptim` |

运行名称记录了两轮训练、batch size 64 和学习率标度 0.0002。SwanLab 中保存的超参数 `config` 为 null；梯度累积、序列长度、种子等其余默认值，根据[记录的 Git 版本对应源码](https://github.com/fumitoshi0524/eiporion-minimind/blob/abad71f5ea0a08d926b21475ae4afa59f57bdb92/trainer/train_pretrain.py)及启动命令 `train_pretrain.py --use_swanlab` 重建。SDK 记录 Git HEAD，并不证明当时工作区没有未提交修改。

控制台记录数据集含 **1,270,238 条样本**，每轮 **19,848 个 microbatch**，与 batch size 64、保留末尾不完整批次相符。两轮共 **39,696 个 microbatch 迭代**；按源码中每四批次更新一次的累积设置，对应 **9,924 次优化器更新**。学习率按 microbatch 推进，而非按优化器更新次数推进。两种步数都不是 token 数；实际非 padding token 计数和数据哈希仍然缺失，因此不能据此确定“40M tokens”的训练预算。

当前公开训练脚本没有调用全局梯度裁剪，所以不再把“梯度裁剪 1.0”列为已实现配置。

| 运行元数据 | 记录值 |
|---|---|
| Run | [`r3jdjca4u4ihm6ei247bc`](https://swanlab.cn/@fumitoshi/Eiporion-Pretrain/runs/r3jdjca4u4ihm6ei247bc) |
| 状态 | Finished |
| GPU | 单张 NVIDIA GeForce RTX 4090 D，容量 24 GB |
| 追踪时间区间，UTC | 2026-05-09 12:26:25 至 15:53:11 |
| 追踪区间长度 | 3 小时 26 分 46 秒 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.7.0 |
| Eiporion | 0.1.0 |
| bitsandbytes / SwanLab | 0.45.5 / 0.7.16 |
| 记录的 Git 版本 | `abad71f5ea0a08d926b21475ae4afa59f57bdb92` |

追踪区间包含启动、日志和检查点等开销，不能当作纯计算时间或 kernel 吞吐率；GPU 容量也不等于训练峰值显存。[精简运行记录](docs/swanlab-run.json) 保留相关环境、计数与控制台证据，不包含无关的主机细节。

## 5. 实验结果

### 5.1 训练轨迹

![根据公开标量历史重绘的训练损失与学习率](img/training-history.svg)

*左图为下一 token 训练损失，右图为学习率。每个序列的全部 398 个记录点均直接绘制，不作平滑；横轴为 microbatch 迭代计数。[原始看板截图](img/train_loss.png) 保留供对照。*

该[公开运行](https://swanlab.cn/@fumitoshi/Eiporion-Pretrain/runs/r3jdjca4u4ihm6ei247bc) 的 `train/loss` 与 `train/lr` 各包含 398 个记录点，标量历史与 413 条控制台日志均对应已完成的两轮训练，轨迹与保存的看板截图一致。

| 观测量 | 记录值 |
|---|---:|
| 首次记录的 loss，microbatch 第 100 步 | 8.093639 |
| 最后记录的 loss，microbatch 第 39,696 步 | 1.968347 |
| 最低记录的 loss，microbatch 第 35,048 步 | 1.829639 |
| 最高记录的学习率，第 8,000 步 | 0.0002 |
| 最后记录的学习率 | 0.00002 |

首次记录点不是初始化 loss，最低采样 loss 也不是验证指标。日志中的 loss 来自被记录的那个 microbatch，不是整轮数据集平均值。这些记录证明该运行的训练目标下降，不能证明泛化或多种子稳定性。

[导出的标量历史](docs/training-history.csv) 可用于直接核对数值。控制台还报告常规 PyTorch 参数 4.93M、INT8 等效权重 58.98M、量化矩阵和 scale 存储 56.49 MiB、整模型等效权重 63.91M，与模型结构和后文核算相符。运行元数据与控制台没有保存检查点 SHA-256 或六项 benchmark 输出，因此具体评测检查点的关联仍未确认。

### 5.2 零样本评测

下表保留项目原实验说明和已有结果汇总中的六项 **zero-shot** 分数，评测工具为 language-model evaluation harness。本次 README 重写未重新运行这些评测；逐样本输出、精确评测版本以及检查点与分数的对应清单尚未随表发布。

| 基准 | Accuracy (%) | Normalized accuracy (%) |
|---|---:|---:|
| ARC-Easy | 28.45 | 28.32 |
| HellaSwag | 27.32 | 28.58 |
| PIQA | 52.23 | 50.82 |
| WinoGrande | 52.64 | - |
| C-Eval，52 科目 | 23.25 | - |
| CMMLU，67 科目 | 25.30 | - |

PIQA 和 WinoGrande 是二选一任务，结果接近均匀随机选择的 50% 参考；HellaSwag、C-Eval 和 CMMLU 接近四选一的 25% 参考。ARC-Easy 不能在未检查实际评测 split 选项数量的情况下，直接指定单一精确随机基线。这些描述不包含不确定性估计，因此不构成统计显著性结论。

这些结果说明该规模和训练配置下的下游能力仍然有限，不能证明量化训练保持了 FP32/BF16 的精度。要识别 INT8 带来的精度代价，需要架构、数据、token 预算和评测设置一致的浮点训练对照；当前没有这样的配对结果。

### 5.3 权重存储

比较采用相同的唯一模型权重，并统一使用 **MiB（$2^{20}$ bytes）**。数值通过保留的本地 `model.safetensors` 导出文件的张量头进行核对，[审计记录](docs/storage-audit.json) 中保留文件 SHA-256 和字节数。检查点本身未上传到仓库，也未独立确认它与上述分数对应的检查点完全一致。

| 存储项 | 元素数 | 每元素字节数 | 存储量（MiB） |
|---|---:|---:|---:|
| INT8 投影权重 | 58,982,400 | 1 | 56.2500 |
| FP32 行 scale | 63,488 | 4 | 0.2422 |
| 共享 FP32 embedding/output 权重 | 4,915,200 | 4 | 18.7500 |
| FP32 归一化权重 | 14,592 | 4 | 0.0557 |
| **权重与 scale 合计** | | | **75.2979** |
| 等效唯一权重全部以 FP32 存储 | 63,912,192 | 4 | **243.8057** |

因此，完整权重表示的存储缩减约为 **3.24 倍**，字节数减少约 **69.1%**。单独比较投影矩阵约有 4 倍缩减，但浮点 embedding/output 权重会降低整模型的压缩比。

完整 safetensors 文件为 78,977,456 bytes，约 75.32 MiB；在权重和 scale 之外，还包含 448 bytes 的运行句柄，以及 21,488 bytes 的文件头和前缀。这些序列化开销不属于模型容量。

早期“约 56.1 MB 对约 245 MB”的比较将 INT8 投影小计与整模型 FP32 估计混用，不能据此宣称完整模型缩减 4.4 倍。以上权重数值也不等于优化器状态、梯度、激活、临时反量化缓冲区、峰值显存或推理吞吐率。

## 6. 讨论与局限

当前最直接的观察是：公开运行的训练 loss 持续下降，其控制台与源码指向采用整数 buffer 的 Transformer 投影，参数统计与模型结构一致；独立检查点核算也展示了明确的权重表示存储优势。因此，它适合作为直接量化更新的小规模实现研究。

仍需回答的核心问题包括：

- **精度成本：** 需要相同训练预算的浮点基线，以隔离 INT8 表示的影响。
- **优化器选择：** 需要比较普通 SR、动量偏置 SR 和确定性取整，不能从单组分数推断优劣。
- **量化粒度：** 实现采用逐行 scale，但没有充分证据支持逐行与逐张量的消融结论。
- **实际效率：** 已核算权重字节数，未测定训练峰值显存、总耗时、tokens/s 或不同硬件上的 kernel 收益。
- **泛化范围：** 单次训练曲线与一组汇总分数，不能证明跨种子、跨规模、长上下文或指令微调的稳定性。

下一步最有价值的实验是：固定数据、精确 token 计数与评测过程，补充浮点基线和优化器消融，将存储优势与可解释的精度、训练成本权衡联系起来。

## 7. 使用代码

### 环境与数据

需要 Python 3.12 或更新版本，依赖见 [pyproject.toml](pyproject.toml)。PyTorch/CUDA 构建应与运行机器匹配。

```bash
git clone https://github.com/fumitoshi0524/eiporion-minimind.git
cd eiporion-minimind
uv sync
```

原训练环境见第 4 节及[运行元数据](docs/swanlab-run.json)。另一份保留的本地环境包含较新的 Eiporion `0.1.1`、Transformers `5.8.0`、SwanLab `0.7.17`，以及 PyTorch `2.6.0+cu124`、lm-eval `0.4.11`，不能用它替代原训练环境来归因结果。仓库未发布锁文件，`uv sync` 可能解析到较新依赖；新实验应保存实际环境和 Eiporion commit。

按 [MiniMind 数据说明](https://github.com/jingyaogong/minimind) 获取 `pretrain_t2t_mini.jsonl`，放入 `dataset/`。数据和训练权重不包含在本仓库中。对照实验应记录数据哈希和非 padding token 总数。

### 训练

```bash
uv run python trainer/train_pretrain.py \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --epochs 2 --batch_size 64 --accumulation_steps 4 \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_length 512 \
  --learning_rate 2e-4 --warmup_steps 8000 --weight_decay 0.1 \
  --from_weight none --save_weight pretrain --use_swanlab
```

训练权重保存到 `out/pretrain_768.pth`，恢复检查点位于 `checkpoints/`。可通过 `--from_resume 1` 继续相应运行。本文分数对应预训练研究，不将仓库中另一个 SFT 脚本视为已有指令微调结果。

### 导出与评测

```bash
uv run python scripts/export_hf.py \
  --weight pretrain --hidden-size 768 --num-hidden-layers 8 --save-dir out

uv run lm_eval --model hf \
  --model_args pretrained=./out,trust_remote_code=True,dtype=float32 \
  --tasks arc_easy,hellaswag,piqa,winogrande,ceval-valid,cmmlu \
  --num_fewshot 0 --batch_size 8 \
  --output_path results/ --log_samples
```

导出过程会包含 Hugging Face 本地加载所需的建模代码和 tokenizer。显式浮点 dtype 保留本文核算中的 FP32 embedding/head 表示，不会把 INT8 buffer 转为浮点主权重。这是启动新评测的配置，不是复现历史分数的保证；应同时保存结果 JSON、任务版本、提示格式、环境与检查点哈希。

无需 GPU、无需加载模型，即可检查导出文件的存储：

```bash
python scripts/inspect_safetensors.py out/model.safetensors
```

### 代码入口

| 路径 | 功能 |
|---|---|
| [`model/model_eiporion.py`](model/model_eiporion.py) | 模型、注意力、FFN 与生成 |
| [`dataset/llm_dataset.py`](dataset/llm_dataset.py) | 分词、padding 与标签 |
| [`trainer/train_pretrain.py`](trainer/train_pretrain.py) | 预训练与优化器接入 |
| [`trainer/train_util.py`](trainer/train_util.py) | 调度、检查点与参数日志 |
| [`scripts/export_hf.py`](scripts/export_hf.py) | Hugging Face 导出 |
| [`scripts/inspect_safetensors.py`](scripts/inspect_safetensors.py) | 仅读取文件头的存储核算 |
| [`docs/storage-audit.json`](docs/storage-audit.json) | 已检查导出的字节数与哈希 |
| [`docs/swanlab-run.json`](docs/swanlab-run.json) | 训练运行的环境、计数与控制台证据 |
| [`docs/training-history.csv`](docs/training-history.csv) | 全部 398 个 loss／学习率记录点 |

## 8. 相关工作与致谢

架构与数据流程建立在 [MiniMind](https://github.com/jingyaogong/minimind) 之上。Eiporion 是作者的配套实现；随机取整与分块优化器量化已有长期研究基础：

- Gupta 等，[Deep Learning with Limited Numerical Precision](https://proceedings.mlr.press/v37/gupta15.html)，ICML 2015：研究低精度训练中的随机取整。
- Dettmers 等，[8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861)，ICLR 2022：研究低精度优化器统计量。Eiporion 对浮点参数使用 bitsandbytes，对 INT8 buffer 采用自己的状态与更新逻辑。
- [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)：提供评测接口；结果与逐样本日志参数见其 [CLI 文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)。

本报告依据[模型代码快照](https://github.com/fumitoshi0524/eiporion-minimind/tree/44f40e5e77786290ae1c962935aa282ac6b0e0b2)、[公开训练运行](https://swanlab.cn/@fumitoshi/Eiporion-Pretrain/runs/r3jdjca4u4ihm6ei247bc)及记录的源码版本、已有评测汇总，以及保留导出的存储元数据编写，分别说明实测观察、理论性质、实现版本差异和尚未建立的实验结论。
