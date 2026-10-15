# Eiporion-MiniMind：小型语言模型的直接 INT8 权重训练

[English report](README.md)

## 摘要

Eiporion-MiniMind 探索一个具体问题：小型 decoder-only 语言模型的投影矩阵，能否在预训练过程中始终以 INT8 格式存储，并直接接受离散权重更新？实现将 MiniMind 衍生的 Transformer 与作者开发的 Eiporion PyTorch 库结合：注意力和前馈网络使用 INT8 权重及逐行 FP32 缩放因子，词嵌入、共享输出头和归一化权重保留浮点表示。

**重点在于训练期间的权重存储：投影矩阵在优化过程中实际保存为 INT8，不另外保留 FP16／FP32 主权重副本。本文实现通过 BF16 矩阵乘法完成训练，不依赖原生 FP4／FP8 训练指令。**

八层模型包含约 63.9M 个等效权重。经过两轮、39,696 个 microbatch 迭代，训练损失从首次记录的 8.0936 降到最后一步的 1.9683。预训练完成后，在本地对模型进行了六项中英文零样本评测。完整权重及缩放因子占 **75.30 MiB**，相同唯一权重的 FP32 表示约占 **243.81 MiB**，存储缩减约 **3.24 倍**。实验建立了可运行的直接量化训练流程，并展示了紧凑权重表示的优势；接近随机选择水平的下游结果，也提示需要进一步通过受控实验研究模型质量。

## 1. 研究问题

低比特训练包含几个独立的设计选择：权重取值所在的量化网格、张量的实际存储 dtype、是否保留浮点主权重，以及矩阵乘法采用的计算精度。仅有低比特前向表示，还不能确定训练中常驻状态的大小。

本项目关注两个实际问题。其一，依靠原生 FP4／FP8 kernel 加速，需要设备和软件栈支持相应算术格式；其二，[BitNet 一类量化感知训练](https://www.jmlr.org/papers/v26/24-2050.html) 虽然采用量化前向权重，仍保留浮点隐含权重来累积更新。消除这一副本，解决的是训练状态预算中的另一部分。

最直接的前作是 Zhao 等人的 [Direct Quantized Training of Language Models with Stochastic Rounding](https://arxiv.org/abs/2412.04787)，研究了不保留高精度隐含权重的直接更新。本项目进一步关注具体实现问题：

> 能否将注意力和前馈投影实际保存在 INT8 张量中，不持有浮点主权重副本，并通过常规 BF16 矩阵乘法完成训练？

困难在于更新分辨率：优化器给出的更新可能远小于整数网格的一步，直接确定性取整会丢失这些更新。实现通过残差累积与随机取整，使较小的更新也有机会改变整数权重。

工作的具体内容包括：

- **底层实现：** 作者开发的 [Eiporion 库](https://github.com/fumitoshi0524/eiporion)，提供 INT8 线性层、自定义反向传播和量化优化器状态。
- **模型集成：** 替换每个 MiniMind 衍生 Transformer block 内的七个投影矩阵，保留浮点嵌入与输出接口。
- **实验评估：** 分析训练动态、六项零样本评测和整模型权重存储。

量化投影矩阵直接接受整数更新，词嵌入、归一化和中间计算采用浮点精度。

| 训练方式 | 常驻权重表示 | 浮点主权重副本 | 矩阵计算 |
|---|---|---|---|
| BitNet 一类 QAT | 浮点隐含权重，前向使用量化值 | 保留 | 混合精度训练 |
| FP8 系统，例如 [FP8-LM](https://arxiv.org/abs/2310.18313) | 按配置组合 FP8 与较高精度状态 | FP8-LM 保留带缩放的较高精度主权重 | 支持 FP8 的 kernel |
| [DQT 参考实现](https://github.com/KYuuto1006/DQT/tree/767a724ed3547b926f785a89613eda00bfa730b3) | 浮点张量承载量化后的取值 | 直接更新过程不另保留隐含权重副本 | 用浮点计算模拟量化网格 |
| **Eiporion-MiniMind** | **INT8 投影 buffer 与 FP32 行 scale** | **量化投影没有主权重副本** | **本文运行使用 BF16 GEMM** |

表格比较代表性实现，并非穷尽所有低精度训练路线；权重表示、计算格式和优化器状态是不同维度。FP8 系统也可以通过压缩其他训练状态减少显存。

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

等效参数统计包含 INT8 投影 buffer 和浮点参数，共享的 embedding/output 权重只计一次：

| 唯一权重类别 | 元素数 |
|---|---:|
| INT8 注意力与 FFN 权重 | 58,982,400 |
| 共享 embedding/output 权重 | 4,915,200 |
| 归一化权重 | 14,592 |
| **模型等效权重总数** | **63,912,192** |
| 额外逐行缩放因子 | 63,488 |

缩放因子属于量化元数据，单独列出；模型等效规模约为 64M。

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
Q_{ij}=\mathrm{clip}\left(\mathrm{round}(W^{(0)}_{ij}/s_i),-127,127\right).
$$

实现对全零行设置很小的正数下限。持久化表示是整数矩阵与行缩放因子，不再为每个量化投影额外持有一份浮点主权重。缩放因子是 buffer，并非由常规 autograd 学习的参数。

本实验使用的 Eiporion 0.1.0 将权重反量化后进行 BF16 矩阵乘法。训练中，投影权重的持久化存储保持 INT8，前向和反向计算则使用浮点有效权重。

### 3.2 自定义反向传播

自定义 autograd 函数根据有效权重计算输入梯度与浮点权重梯度：

$$
G_X=G_Y W,\qquad G_W=G_Y^\top X.
$$

权重梯度累积到对应模块的缓存，再由优化器取出以更新整数 buffer。这解决了 INT8 张量无法作为普通浮点可训练参数处理的问题。

得到的浮点梯度为离散权重提供更新方向，再由残差累积与取整决定这一方向如何改变整数值。

### 3.3 优化器、残差与随机取整

`EiporionOptim` 将 AdamW 风格更新与分块量化的一阶矩、二阶矩和更新残差结合。INT8 路径的块大小为 256，使用逐块缩放和非线性压扩；剩余浮点参数由底层 bitsandbytes 优化器处理。

在固定行缩放因子下，可将更新简化为：

$$
u_t=-\eta_t\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda W_t\right),\qquad
r_t\leftarrow r_{t-1}+u_t/s,
$$

$$
\Delta Q_t=\mathcal{R}(r_t),\qquad
Q_{t+1}=\mathrm{clip}(Q_t+\Delta Q_t,-127,127),\qquad
r_t\leftarrow r_t-\Delta Q_t.
$$

除以 $s$ 是逐行操作。残差保留尚未施加到整数权重上的更新分量，并在优化器步之间以分块量化形式存储。

普通随机取整依据小数部分，在相邻整数间进行概率选择，未发生裁剪时满足 $\mathbb{E}[\mathcal{R}(z)\mid z]=z$。它使小于一个整数步长的更新仍能影响整数权重。

本实验通过 `EiporionOptim` 使用**动量偏置随机取整**，`sr_bias_scale=0.15`：在小数取整概率中加入自适应动量项，再裁剪到 $[0,1]$。库中另有 `EiporionOptimSR` 对应普通随机取整。上述无偏性质描述普通变体；带偏置的规则会主动改变更新分布。

本实验采用的 [Eiporion 0.1.0](https://github.com/fumitoshi0524/eiporion/tree/bff6aab67a061b4a006b6d5a1dc6908b7c41c438) 在整数更新过程中保持行 scale 固定。

### 3.4 训练期间的常驻状态

每个量化投影在优化器步之间保留 `torch.int8` 矩阵和 FP32 行 scale，没有对应的浮点 `nn.Parameter` 作为主权重矩阵。一阶矩、二阶矩和更新残差以分块 8-bit 形式保存，浮点 embedding/output 与归一化参数单独优化。

计算过程中仍会生成 BF16 有效权重张量、缓存权重梯度，优化器处理量化状态时也需要浮点工作张量。这些计算／更新缓冲区与常驻的主权重副本不同。因此，紧凑权重表示在训练期间已经存在；它对峰值 GPU 显存的影响，还取决于这些缓冲区及激活的占用。

## 4. 训练设置

预训练使用 MiniMind 的 `pretrain_t2t_mini.jsonl`。数据加载器读取每条样本的 `text` 字段，进行分词、截断、添加首尾标记并补齐到 512 tokens；padding 对应的 label 设置为 `-100`，模型内部移位后计算下一 token 的交叉熵。

| 设置 | 数值 |
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

数据集包含 **1,270,238 条样本**，每轮 **19,848 个 microbatch**，包含末尾不完整批次。两轮共 **39,696 个 microbatch 迭代**，在四步梯度累积设置下对应 **9,924 次优化器更新**。学习率调度按 microbatch 推进。

| 运行环境 | 数值 |
|---|---|
| GPU | 单张 NVIDIA GeForce RTX 4090 D，容量 24 GB |
| 端到端运行时长 | 约 3 小时 27 分钟 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.7.0 |
| Eiporion | 0.1.0 |
| bitsandbytes / SwanLab | 0.45.5 / 0.7.16 |

运行时长包含数据加载、日志和检查点保存。

## 5. 实验结果

### 5.1 训练轨迹

![根据公开标量历史重绘的训练损失与学习率](img/training-history.svg)

*左图为两轮训练中的下一 token 损失，右图为学习率。横轴为 microbatch 迭代计数，曲线使用日志原始值，不作平滑。*

训练损失在预热前期快速下降，随后随着学习率衰减继续缓慢降低，在第二轮结束时接近 2。学习率在 microbatch 第 8,000 步达到峰值，最终下降到峰值的十分之一。

| 观测量 | 记录值 |
|---|---:|
| 首次记录的 loss，microbatch 第 100 步 | 8.093639 |
| 最后记录的 loss，microbatch 第 39,696 步 | 1.968347 |
| 最低记录的 loss，microbatch 第 35,048 步 | 1.829639 |
| 最高记录的学习率，第 8,000 步 | 0.0002 |
| 最后记录的学习率 | 0.00002 |

表中 loss 来自被记录的训练 microbatch。该轨迹表明，在这组配置下，离散投影更新能够有效优化下一 token 预测目标。

### 5.2 零样本评测

预训练完成后，使用 [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) 在**本地**进行了六项中英文评测，均采用 **zero-shot** 设置，不作任务特定微调。下表列出 accuracy，以及适用任务的 normalized accuracy。

| 基准 | Accuracy (%) | Normalized accuracy (%) |
|---|---:|---:|
| ARC-Easy | 28.45 | 28.32 |
| HellaSwag | 27.32 | 28.58 |
| PIQA | 52.23 | 50.82 |
| WinoGrande | 52.64 | - |
| C-Eval，52 科目 | 23.25 | - |
| CMMLU，67 科目 | 25.30 | - |

结果表明下游能力仍然有限：PIQA 和 WinoGrande 接近二选一的 50% 参考，HellaSwag、C-Eval 和 CMMLU 接近四选一的 25% 参考。训练目标的降低尚未转化为这些任务上的较强表现。

### 5.3 权重存储

比较覆盖相同的唯一模型权重，并统一使用 **MiB（$2^{20}$ bytes）**。INT8 表示的总量包含行 scale、浮点 embedding/output 以及归一化权重。

| 存储项 | 元素数 | 每元素字节数 | 存储量（MiB） |
|---|---:|---:|---:|
| INT8 投影权重 | 58,982,400 | 1 | 56.2500 |
| FP32 行 scale | 63,488 | 4 | 0.2422 |
| 共享 FP32 embedding/output 权重 | 4,915,200 | 4 | 18.7500 |
| FP32 归一化权重 | 14,592 | 4 | 0.0557 |
| **权重与 scale 合计** | | | **75.2979** |
| 等效唯一权重全部以 FP32 存储 | 63,912,192 | 4 | **243.8057** |

因此，完整权重表示的存储缩减约为 **3.24 倍**，字节数减少约 **69.1%**。单独比较投影矩阵约有 4 倍缩减，但浮点 embedding/output 权重会降低整模型的压缩比。

完整 safetensors 文件包含序列化元数据后约为 75.32 MiB。以上表格衡量权重表示的存储成本，实际训练还需容纳优化器状态、梯度、激活和临时计算缓冲区。

相同的 INT8 buffer 也承担训练期间的常驻投影权重，因此表示上的节省在导出前就已存在。3.24 倍量化的是权重与 scale 的存储缩减，不代表训练峰值显存的整体缩减比例。

## 6. 讨论与局限

本实验建立了整数存储投影权重的实际训练流程，将整模型权重存储减少约 69%。下一步的核心问题是：这种存储优势与模型质量之间有怎样的关系。

仍需回答的核心问题包括：

- **模型质量：** 需要配对的浮点训练对照来隔离 INT8 训练的影响；接近随机水平的选择题分数，对这种比较的区分度有限。
- **更新设计：** 对普通 SR、动量偏置 SR 与确定性取整进行受控比较，可以明确优化器的作用；逐行与逐张量 scale 也是有价值的消融方向。
- **系统效率与规模：** 进一步测量峰值训练显存、吞吐率，并在更大模型上重复实验，才能判断权重存储节省如何转化为实际训练收益。

下一步最有价值的实验是：固定数据、精确 token 计数与评测过程，补充浮点基线和优化器消融，将存储优势与可解释的精度、训练成本权衡联系起来。

## 7. 使用代码

### 环境与数据

需要 Python 3.12 或更新版本，依赖见 [pyproject.toml](pyproject.toml)。PyTorch/CUDA 构建应与运行机器匹配。

```bash
git clone https://github.com/fumitoshi0524/eiporion-minimind.git
cd eiporion-minimind
uv sync
```

本报告实验采用第 4 节所列环境，包括 Eiporion 0.1.0。新运行应随训练配置一并记录实际依赖版本。

按 [MiniMind 数据说明](https://github.com/jingyaogong/minimind) 获取 `pretrain_t2t_mini.jsonl`，放入 `dataset/`。训练数据和模型权重与本代码仓库分开存放。

### 训练

```bash
uv run python trainer/train_pretrain.py \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --epochs 2 --batch_size 64 --accumulation_steps 4 \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_length 512 \
  --learning_rate 2e-4 --warmup_steps 8000 --weight_decay 0.1 \
  --from_weight none --save_weight pretrain --use_swanlab
```

训练权重保存到 `out/pretrain_768.pth`，恢复检查点位于 `checkpoints/`。可通过 `--from_resume 1` 继续已有运行。

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

导出过程包含 Hugging Face 本地加载所需的建模代码和 tokenizer。显式浮点 dtype 保留 FP32 embedding/head 表示，投影 buffer 继续使用 INT8。评测结果与逐样本预测保存到 `results/`。

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

## 8. 相关工作与致谢

架构与数据流程建立在 [MiniMind](https://github.com/jingyaogong/minimind) 之上，Eiporion 是作者的配套实现。最相关的研究涉及直接量化更新、隐含权重 QAT、低精度计算和优化器状态存储。

### 直接量化更新

- **Zhao 等，[Direct Quantized Training of Language Models with Stochastic Rounding](https://arxiv.org/abs/2412.04787)，ACML 2025**（[代码](https://github.com/KYuuto1006/DQT)）。这是最直接的方法前作，通过随机取整实现不持有额外高精度隐含矩阵的更新。其公开[线性层](https://github.com/KYuuto1006/DQT/blob/767a724ed3547b926f785a89613eda00bfa730b3/utils_quant.py) 继承 `nn.Linear`，[取整实现](https://github.com/KYuuto1006/DQT/blob/767a724ed3547b926f785a89613eda00bfa730b3/custom_optimizer.py) 将结果转回输入 dtype。Eiporion 显式以 INT8 buffer 保存投影矩阵，并结合分块量化优化器状态执行更新。

### 隐含权重与计算精度

- **Wang 等，[BitNet: 1-bit Pre-training for Large Language Models](https://www.jmlr.org/papers/v26/24-2050.html)，JMLR 2025。** BitNet 展示了大规模低比特语言模型训练，其混合精度方案保留高精度隐含权重来累积更新，并在前向时量化。Eiporion 则直接对整数 buffer 施加投影权重更新。
- **Peng 等，[FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313)，2023。** FP8-LM 组合低精度计算、通信和优化器状态，在 H100 上报告了加速效果；其默认优化器仍保留带缩放的 FP16 主权重。这与通过整数 buffer 和 BF16 计算降低常驻权重存储是互补的系统设计方向。
- **Zamirai 等，[Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192)，2020／2021。** 研究 BF16 训练中小更新的随机取整与补偿求和，为减少对高精度更新累积的依赖提供了更广泛的背景。

### 数值方法与优化器基础

- Gupta 等，[Deep Learning with Limited Numerical Precision](https://proceedings.mlr.press/v37/gupta15.html)，ICML 2015：研究低精度训练中的随机取整。
- Dettmers 等，[8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861)，ICLR 2022：研究低精度优化器统计量。Eiporion 对浮点参数使用 bitsandbytes，对 INT8 buffer 采用自己的状态与更新逻辑。
- [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)：提供评测接口；结果与逐样本日志参数见其 [CLI 文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)。

<details>
<summary>实验记录</summary>

- [SwanLab 训练运行](https://swanlab.cn/@fumitoshi/Eiporion-Pretrain/runs/r3jdjca4u4ihm6ei247bc)
- [Loss 与学习率历史](docs/training-history.csv)
- [训练环境与运行元数据](docs/swanlab-run.json)
- [权重存储明细](docs/storage-audit.json)
- [训练源码版本](https://github.com/fumitoshi0524/eiporion-minimind/tree/abad71f5ea0a08d926b21475ae4afa59f57bdb92)

</details>
