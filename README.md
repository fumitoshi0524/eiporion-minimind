# Eiporion-MiniMind: Direct INT8 Weight Training for a Small Language Model

[中文报告](README_zh.md)

## Try the model

**[Open the browser playground ↗](https://fumitoshi0524.github.io/eiporion-minimind/?lang=en)** · [Weights and tokenizer](demo/model/) · [Generation script](scripts/generate.py)

A roughly 64M-parameter model with a **75.32 MiB** checkpoint. Click **Load model** to download the weights and generate text on your own device. The page uses WebGPU when available and automatically falls back to a CPU worker otherwise. CPU generation is slower. No login or API key is needed; both backends preserve INT8 projection storage and use floating-point computation.

The release is a **pretrained checkpoint**. Enter the beginning of a passage and explore its continuation; this checkpoint has not been instruction-tuned for chat.

### An actual generation

**Prompt:** Artificial intelligence is

**Actual browser continuation:**

> becoming increasingly important in any single, as of the latest industry, we are excellent, technology, and artificial intelligence. It is crucial to remember that AI is used in various fields such as computer science, artificial intelligence, and other applications. It is important to ensure that

An unedited English sample generated with the browser CPU backend linked above (64-token limit). To generate locally with PyTorch after cloning the repository:

```bash
uv sync
uv run python scripts/generate.py --model-dir demo/model \
  --prompt "Artificial intelligence is" --max-new-tokens 96 --seed 42
```

## Abstract

Eiporion-MiniMind investigates whether the projection weights of a small decoder-only language model can remain in INT8 storage while being updated during pretraining. The implementation combines a MiniMind-derived transformer with Eiporion, a companion PyTorch library developed for quantized linear layers and discrete weight updates. Attention and feed-forward projections use INT8 weight buffers and per-row floating-point scales; the shared token embedding/output head and normalization weights remain floating point.

**The focus is training-time weight storage: projection matrices are physically stored as INT8 throughout optimization, without separate FP16/FP32 master-weight copies. The reported implementation uses BF16 matrix multiplication and does not require native FP4/FP8 training instructions.**

The eight-layer model has approximately 63.9 million dense-equivalent weights. Over two epochs and 39,696 microbatch iterations, training loss decreases from 8.0936 at the first logged step to 1.9683 at the final step. Its weights and scales occupy **75.30 MiB**, compared with **243.81 MiB for the equivalent FP32 representation**, a **3.24× storage reduction**. The experiment establishes a working direct quantized training pipeline. The released weights can be used locally or explored through an in-browser text-completion playground.

## 1. Research question and scope

Low-bit training involves several design choices: the numerical grid of the weights, their physical storage dtype, the use of a floating-point master copy, and the arithmetic used for matrix multiplication. A low-bit forward representation alone does not determine the amount of state resident during training.

Two practical considerations motivate this project. First, acceleration through native FP4/FP8 kernels depends on the arithmetic supported by the device and software stack. Second, [BitNet-style quantization-aware training](https://www.jmlr.org/papers/v26/24-2050.html) retains floating-point latent weights for updates, even though its forward weights are quantized. Removing that latent copy addresses a separate part of the training-state budget.

The closest methodological precedent is Zhao et al.'s [Direct Quantized Training of Language Models with Stochastic Rounding](https://arxiv.org/abs/2412.04787), which studies direct updates without high-precision latent weights. Here, we investigate a concrete implementation question:

> Can attention and feed-forward projections be maintained as actual INT8 tensors, updated without floating-point master copies, and trained through ordinary BF16 matrix multiplication?

The main difficulty is update resolution. A small adaptive-optimizer update can be much smaller than one step on an integer grid; deterministic rounding can erase it. The implementation combines accumulated update residuals with stochastic rounding so that these small updates can eventually change the stored integer weights.

The work comprises three connected pieces:

- **Eiporion:** the author's [companion library](https://github.com/fumitoshi0524/eiporion) for INT8 linear layers, custom backward computation, and quantized optimizer state.
- **Model integration:** replacing the seven projection matrices in each MiniMind-derived transformer block while retaining a floating-point embedding/output interface.
- **Runnable model:** training curves, whole-model storage analysis, public weights, and browser-local text generation.

The quantized projection matrices are updated directly; embeddings, normalization, and intermediate computation use floating-point precision.

| Training approach | Persistent weight representation | Floating-point master copy | Matrix computation |
|---|---|---|---|
| BitNet-style QAT | Floating-point latent weights with quantized forward weights | Retained | Mixed-precision training |
| FP8 training systems, e.g. [FP8-LM](https://arxiv.org/abs/2310.18313) | FP8 and higher-precision state, depending on the recipe | FP8-LM retains a scaled higher-precision master | FP8-capable kernels |
| [DQT reference implementation](https://github.com/KYuuto1006/DQT/tree/767a724ed3547b926f785a89613eda00bfa730b3) | Quantized values represented by floating-point tensors | No separate latent-weight copy in the direct update | Floating-point simulation of the quantized grid |
| **Eiporion-MiniMind** | **INT8 projection buffers with FP32 row scales** | **No master copy for quantized projections** | **BF16 GEMM in the reported run** |

These are representative implementations, rather than an exhaustive classification. The comparison separates weight representation from arithmetic and optimizer state; FP8 systems can also reduce training memory through other components.

## 2. Model architecture

The model follows the decoder-only design used in [MiniMind](https://github.com/jingyaogong/minimind): grouped-query attention, rotary position embeddings, RMSNorm, and SwiGLU feed-forward blocks. QK normalization is applied before attention-score computation.

| Component | Configuration |
|---|---|
| Transformer blocks | 8 |
| Hidden dimension | 768 |
| Query / key-value heads | 8 / 4 |
| Head dimension | 96 |
| Feed-forward intermediate dimension | 2,432 |
| Vocabulary | 6,400 tokens |
| Dropout | 0 |
| Embedding and output head | Shared floating-point weight matrix |
| Quantized projections per block | Q, K, V, O, gate, up, down |
| Stored projection weights | INT8 buffers with FP32 row scales |
| Configured positional buffer | 32,768 positions |
| Pretraining sequence length | 512 tokens |

![Architecture of the MiniMind-derived model](img/model-and-update.svg)

*Left: model structure. Right: the update loop for one quantized projection. Green denotes compact persistent state; gray denotes floating-point computation.*

The 32,768-position configuration is not an evaluated context length. The code includes optional YaRN scaling, but `inference_rope_scaling` is disabled by default and no long-context evaluation is reported here.

### Parameter counting

The dense-equivalent count includes both INT8 projection buffers and floating-point parameters, with the shared embedding/output weight counted once.

For the configuration above:

| Unique weight group | Scalar count |
|---|---:|
| INT8 attention and feed-forward weights | 58,982,400 |
| Shared embedding/output weight | 4,915,200 |
| Normalization weights | 14,592 |
| **Dense-equivalent model weights** | **63,912,192** |
| Additional row-scale values | 63,488 |

The scales are quantization metadata and are listed separately. This gives the approximately 64M model size used throughout this report.

These counts can also be derived without a checkpoint. With $h=768$, $k=384$ (four KV heads), $f=2432$, $L=8$, and $V=6400$, the projection count is $L(2h^2+2hk+3hf)$; the shared embedding count is $Vh$; normalization contributes $L(2h+2\times96)+h$ values. The number of row scales is $L(3h+2k+2f)$.

## 3. Quantized representation and optimization

### 3.1 INT8 linear layers

For an output-by-input integer matrix $Q$ and a scale vector $s$, the effective weight is

$$
W_{ij}=s_i Q_{ij},\qquad Q_{ij}\in\{-127,\ldots,127\}.
$$

With inputs represented as rows, the layer computes $Y=XW^\top$. The transformer projections use no bias. Initialization starts from a Kaiming-uniform floating-point matrix and uses symmetric per-row scaling:

$$
s_i=\frac{\max_j |W^{(0)}_{ij}|}{127},\qquad
Q_{ij}=\mathrm{clip}\!\left(\mathrm{round}(W^{(0)}_{ij}/s_i),-127,127\right),
$$

with a small positive floor for zero rows. After initialization, the persistent representation is the integer matrix and its row scales rather than a separate floating-point master matrix for each quantized projection. The scales are buffers, not parameters learned by ordinary autograd.

For this experiment, Eiporion 0.1.0 dequantizes weights for BF16 matrix multiplication. The persistent projection storage remains INT8 during training, while forward and backward computation operate on floating-point effective weights.

### 3.2 Custom backward computation

The custom autograd function computes input gradients and a floating-point gradient with respect to the effective weight:

$$
G_X=G_Y W,\qquad G_W=G_Y^\top X.
$$

It accumulates the weight gradient in a module-specific cache. The optimizer consumes that cache to update the INT8 buffer, which PyTorch cannot treat as a conventional differentiable floating-point parameter.

The resulting floating-point gradient supplies the update direction for the discrete weights; residual accumulation and rounding determine how that direction changes the stored integers.

### 3.3 Adaptive updates, residuals, and rounding

`EiporionOptim` combines AdamW-style updates with blockwise quantized first moments, second moments, and update residuals. The INT8 path uses blocks of 256 values with per-block scales and nonlinear companding. The remaining floating-point parameters are handled by the underlying bitsandbytes optimizer.

For a fixed row scale, a simplified description is:

$$
u_t=-\eta_t\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda W_t\right),\qquad
r_t\leftarrow r_{t-1}+u_t/s,
$$

$$
\Delta Q_t=\mathcal{R}(r_t),\qquad
Q_{t+1}=\mathrm{clip}(Q_t+\Delta Q_t,-127,127),\qquad
r_t\leftarrow r_t-\Delta Q_t.
$$

The division by $s$ is rowwise. Residuals retain update components that have not yet been applied to the integer weights, and are stored in blockwise quantized form between steps.

For ordinary stochastic rounding, a scalar $z$ is rounded to $\lfloor z\rfloor$ or $\lceil z\rceil$ with probabilities determined by its fractional part, giving $\mathbb{E}[\mathcal{R}(z)\mid z]=z$ before clipping. This explains how sub-integer updates can influence integer weights.

The experiment uses **momentum-biased stochastic rounding** through `EiporionOptim`, with `sr_bias_scale=0.15`. This adds an adaptive-momentum term to the fractional rounding probability before clipping it to $[0,1]$. The library also exposes `EiporionOptimSR` for ordinary stochastic rounding. The unbiasedness identity above describes that ordinary variant; the biased rule deliberately changes the update distribution.

In [Eiporion 0.1.0](https://github.com/fumitoshi0524/eiporion/tree/bff6aab67a061b4a006b6d5a1dc6908b7c41c438), the row scales remain fixed during these integer updates.

### 3.4 State resident during training

Each quantized projection retains its `torch.int8` matrix and FP32 row scales between optimization steps. There is no corresponding floating-point `nn.Parameter` serving as a master matrix. Its first moment, second moment, and update residual are stored in blockwise 8-bit form. Floating-point embedding/output and normalization parameters are optimized separately.

BF16 effective-weight tensors and cached weight gradients are still materialized for computation, and the optimizer uses floating-point working tensors when processing its quantized state. These are compute/update buffers, distinct from a persistent master-weight copy. The compact representation therefore applies during training as well as checkpoint storage, while its effect on peak GPU allocation also depends on these buffers and activations.

## 4. Training setup

Pretraining uses the MiniMind `pretrain_t2t_mini.jsonl` dataset and next-token cross-entropy. The dataset loader reads a `text` field, tokenizes and truncates each example, adds beginning/end tokens, and pads to 512 tokens. Padding labels are masked with `-100`; the model shifts logits and labels for next-token prediction.

| Setting | Value |
|---|---|
| Epochs | 2 |
| Microbatch size | 64 sequences |
| Gradient accumulation | 4 microbatches |
| Nominal effective batch size | 256 sequences on one device |
| Sequence length | 512 |
| Learning-rate scale | $2\times10^{-4}$ |
| Warmup | 8,000 microbatch iterations |
| Schedule | Linear warmup from 10% of the LR scale, followed by cosine decay toward 10% |
| Weight decay | 0.1 |
| Autocast precision | BF16 |
| Initial seed | 42; data-order seed changes by epoch |
| Projection optimizer | `EiporionOptim` |

The dataset contains **1,270,238 examples**. Each epoch comprises **19,848 microbatches**, including the final partial batch. Two epochs therefore give **39,696 microbatch iterations** and **9,924 optimizer updates** with four-step accumulation. The learning-rate schedule advances per microbatch.

| Runtime environment | Value |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 4090 D, 24 GB capacity |
| End-to-end run duration | Approximately 3 h 27 min |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.7.0 |
| Eiporion | 0.1.0 |
| bitsandbytes / SwanLab | 0.45.5 / 0.7.16 |

The run duration includes data loading, logging, and checkpointing.

## 5. Results

### 5.1 Training behavior

![Training loss and learning rate redrawn from the public scalar history](img/training-history.svg)

*Next-token training loss and learning rate over two epochs. The horizontal axis counts microbatch iterations; curves show logged values without smoothing.*

Training loss falls rapidly during the early warmup phase and then decreases more gradually as the learning rate decays. It reaches approximately 2 by the end of the second epoch. The learning rate peaks at microbatch step 8,000 and decays to one tenth of that peak.

| Observed quantity | Value |
|---|---:|
| First logged loss, microbatch step 100 | 8.093639 |
| Final logged loss, microbatch step 39,696 | 1.968347 |
| Lowest logged loss, microbatch step 35,048 | 1.829639 |
| Peak logged learning rate, microbatch step 8,000 | 0.0002 |
| Final logged learning rate | 0.00002 |

Loss values are measured on the logged training microbatches. The trajectory shows that the discrete projection updates support optimization of the next-token objective in this configuration.

### 5.2 Weight-storage accounting

Weight storage is calculated over the same unique model weights in each representation, using **MiB ($2^{20}$ bytes)** throughout. The INT8 representation includes its row scales and the floating-point embedding/output and normalization weights.

| Stored component | Values | Bytes per value | Storage (MiB) |
|---|---:|---:|---:|
| INT8 projection weights | 58,982,400 | 1 | 56.2500 |
| FP32 row scales | 63,488 | 4 | 0.2422 |
| Shared FP32 embedding/output weight | 4,915,200 | 4 | 18.7500 |
| FP32 normalization weights | 14,592 | 4 | 0.0557 |
| **Weights and scales** | | | **75.2979** |
| Equivalent unique weights in FP32 | 63,912,192 | 4 | **243.8057** |

Thus,

$$
\frac{243.8057}{75.2979}\approx3.24,
$$

or approximately **69.1% fewer bytes** for weights and scales relative to the FP32 representation. Compressing only the projection matrices yields approximately 4× storage reduction for those matrices, but the floating-point embedding/output weight reduces the whole-model ratio.

The serialized safetensors checkpoint occupies approximately 75.32 MiB including file metadata. The table measures weight representation cost; training memory additionally includes optimizer state, gradients, activations, and temporary compute buffers.

The same INT8 buffers are the persistent projection weights during training. Thus, the representation saving is present before export; the 3.24× figure specifically quantifies weights and scales, rather than total peak training memory.

## 6. Discussion and limitations

The experiment demonstrates a practical pipeline for training integer-stored transformer projections and reduces whole-model weight storage by approximately 69%. Its main unresolved question is the relationship between that storage advantage and model quality.

Several questions remain open:

- **Model quality:** a matched floating-point run is needed to isolate the effect of INT8 training. The released checkpoint is pretrained only and is presented as text completion.
- **Update design:** controlled comparisons of ordinary SR, momentum-biased SR, and deterministic rounding would clarify the role of the optimizer; per-row and per-tensor scales are another useful ablation.
- **System efficiency and scale:** peak training memory, throughput, and repeated runs at larger model sizes would establish how the weight-storage savings translate into practical training benefits.

The next useful experiment is a controlled floating-point baseline and optimizer ablation with identical data and logged token counts. This would connect the implementation's storage advantage to an interpretable accuracy and training-cost trade-off.

## 7. Using the implementation

### Environment and data

Python 3.12 or newer is required. The project declares its dependencies in [pyproject.toml](pyproject.toml); CUDA/PyTorch compatibility must match the machine running the experiment.

```bash
git clone https://github.com/fumitoshi0524/eiporion-minimind.git
cd eiporion-minimind
uv sync
```

The reported experiment uses the environment in Section 4, including Eiporion 0.1.0. Package versions for a new run should be recorded alongside its training configuration.

Obtain `pretrain_t2t_mini.jsonl` using the [MiniMind data instructions](https://github.com/jingyaogong/minimind), and place it in `dataset/`. The released pretrained weights and tokenizer are in [`demo/model/`](demo/model/); training data must be obtained separately.

### Train

```bash
uv run python trainer/train_pretrain.py \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --epochs 2 --batch_size 64 --accumulation_steps 4 \
  --hidden_size 768 --num_hidden_layers 8 --max_seq_length 512 \
  --learning_rate 2e-4 --warmup_steps 8000 --weight_decay 0.1 \
  --from_weight none --save_weight pretrain --use_swanlab
```

The trainer writes `out/pretrain_768.pth` and resume checkpoints under `checkpoints/`. Use `--from_resume 1` to continue a saved run.

### Export and generate

```bash
uv run python scripts/export_hf.py \
  --weight pretrain --hidden-size 768 --num-hidden-layers 8 --save-dir out

uv run python scripts/generate.py --model-dir out \
  --prompt "Artificial intelligence is" --max-new-tokens 128
```

The export includes configuration, weights, and tokenizer files for local loading. The generation script preserves FP32 embedding/head weights and INT8 projection buffers. Completion is the default; `--mode chat` is reserved for a future instruction-tuned checkpoint.

For an inspection of storage without loading the model or requiring a GPU:

```bash
python scripts/inspect_safetensors.py out/model.safetensors
```

### Repository map

| Path | Purpose |
|---|---|
| [`model/model_eiporion.py`](model/model_eiporion.py) | Model, attention, feed-forward blocks, and generation |
| [`dataset/llm_dataset.py`](dataset/llm_dataset.py) | Tokenization, padding, and training labels |
| [`trainer/train_pretrain.py`](trainer/train_pretrain.py) | Pretraining loop and optimizer wiring |
| [`trainer/train_util.py`](trainer/train_util.py) | Schedule, checkpointing, and parameter logging |
| [`scripts/export_hf.py`](scripts/export_hf.py) | Hugging Face checkpoint export |

## 8. Related work and acknowledgements

The architecture and data pipeline build on [MiniMind](https://github.com/jingyaogong/minimind). Eiporion is the author's companion implementation. The most relevant research concerns direct quantized updates, latent-weight QAT, low-precision arithmetic, and optimizer-state storage.

### Direct quantized updates

- **Zhao et al., [Direct Quantized Training of Language Models with Stochastic Rounding](https://arxiv.org/abs/2412.04787), ACML 2025** ([code](https://github.com/KYuuto1006/DQT)). DQT studies updates directly on a quantization grid, using stochastic rounding to preserve small updates. Eiporion shares this numerical background and implements an **integer-storage training path spanning weight representation, backward computation, and persistent optimizer state**.

Four implementation choices work together:

| Component | DQT public reference implementation | Eiporion implementation |
|---|---|---|
| Weight container | Floating-point `nn.Linear` parameters carry grid values; rounding returns the input dtype | Actual `torch.int8` buffers plus FP32 row scales; 58.98M projection weights occupy 56.25 MiB |
| Gradient/update interface | Floating-point parameter gradients and optimizer interface | Custom backward caches effective-weight gradients; the optimizer updates integer buffers |
| Sub-grid updates | Stochastic rounding of direct weight updates | Residual accumulation in integer coordinates, momentum-biased stochastic rounding, and carry-over of the remaining update |
| Persistent optimizer state | The cited `MyAdamW` path initializes floating-point moments | First moments, second moments, and update residuals are persisted in blockwise 8-bit form |

The distinction spans more than a rounding function: physical integer weights, residual updates, and quantized optimizer state are integrated into one training pipeline and a language model that can be trained, exported, and run interactively. BF16 GEMM performs the floating-point computation without requiring native FP4/FP8 training instructions. This comparison refers to the cited [DQT linear layer](https://github.com/KYuuto1006/DQT/blob/767a724ed3547b926f785a89613eda00bfa730b3/utils_quant.py) and [optimizer](https://github.com/KYuuto1006/DQT/blob/767a724ed3547b926f785a89613eda00bfa730b3/custom_optimizer.py); quantized optimizer-state storage is not the main question addressed by that paper.

### Latent weights and arithmetic precision

- **Wang et al., [BitNet: 1-bit Pre-training for Large Language Models](https://www.jmlr.org/papers/v26/24-2050.html), JMLR 2025.** BitNet establishes low-bit language-model training at scale. Its mixed-precision recipe retains high-precision latent weights to accumulate updates and quantizes them for the forward pass. Eiporion instead applies the projection updates to integer buffers.
- **Peng et al., [FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313), 2023.** FP8-LM combines low-precision computation, communication, and optimizer state, with its reported acceleration on H100 hardware. Its default optimizer still uses scaled FP16 master weights. This is complementary to reducing persistent weight storage through integer buffers and BF16 computation.
- **Zamirai et al., [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192), 2020/2021.** This work studies stochastic rounding and compensated summation for small updates in BF16 training, providing broader context for reducing dependence on high-precision update accumulation.

### Numerical and optimizer foundations

- Gupta et al., [*Deep Learning with Limited Numerical Precision*](https://proceedings.mlr.press/v37/gupta15.html), ICML 2015, studies stochastic rounding for low-precision training.
- Dettmers et al., [*8-bit Optimizers via Block-wise Quantization*](https://arxiv.org/abs/2110.02861), ICLR 2022, develops low-precision optimizer statistics. Eiporion uses bitsandbytes for its floating-point parameters and its own state/update logic for INT8 buffers.

<details>
<summary>Experiment records</summary>

- [Training run on SwanLab](https://swanlab.cn/@fumitoshi/Eiporion-Pretrain/runs/r3jdjca4u4ihm6ei247bc)
- [Loss and learning-rate history](docs/training-history.csv)
- [Training environment and run metadata](docs/swanlab-run.json)
- [Weight-storage breakdown](docs/storage-audit.json)
- [Training source revision](https://github.com/fumitoshi0524/eiporion-minimind/tree/abad71f5ea0a08d926b21475ae4afa59f57bdb92)

</details>
