# Browser-local text completion

This static page runs the released pretrained Eiporion-MiniMind checkpoint on the visitor's device, using WebGPU where available and a CPU compatibility backend otherwise. No inference server, API key, or chat service is involved. The checkpoint has not been instruction-tuned.

- `webgpu.js`: safetensors loading, packed INT8 projection matrix–vector products, FP32 row scales, RMSNorm, QK normalization, RoPE, grouped-query attention with a KV cache, SwiGLU, and the shared floating-point output head.
- `cpu.js`: integer-weight matrix–vector operations and cached autoregressive decoding for browsers without a WebGPU adapter; runs inside the worker.
- `weights.js`: range-aware checkpoint loading for the CPU backend.
- `worker.js`: tokenizer loading, prompt processing, top-k/top-p sampling, and streaming responses.
- `model/`: the 75.32 MiB checkpoint, configuration, and tokenizer.
- `vendor/transformers.min.js`: Hugging Face Transformers.js 3.8.1, used for tokenization only. Its Apache 2.0 license is included in `vendor/LICENSE`. No ONNX inference runtime is used by this page.

Projection weights are read directly from the original INT8 safetensors buffers. Browser arithmetic is FP32 rather than the BF16 arithmetic used for the reported training run, so generated samples need not match PyTorch outputs. Both prompt processing and generation are sequential token decoding, with a maximum combined context of 512 tokens. WebGPU requires a secure context and an available adapter. Initialization tries the high-performance, default, low-power, and compatibility adapters before switching to CPU. GPU access can be unavailable inside embedded browsers or when hardware acceleration is disabled; CPU mode needs neither a GPU adapter nor browser flags. Use `?lang=en` for the English interface and `?lang=zh` for Chinese.

Serve this directory over localhost for local use, or deploy it as static assets. The repository's Pages workflow publishes it automatically. The model download supports HTTP range requests and falls back to a full response when the server does not support ranges.
