"""Generate text from a local Eiporion Hugging Face export."""
import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer, TextStreamer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.model_eiporion import EiporionConfig, EiporionForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("out"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=["completion", "chat"], default="completion")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=Path)
    args = parser.parse_args()
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p must be in (0, 1].")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive.")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    config = EiporionConfig.from_pretrained(args.model_dir, local_files_only=True)
    model = EiporionForCausalLM(config)
    load_model(model, str(args.model_dir / "model.safetensors"), strict=True)
    model = model.eval().to(args.device)
    if args.mode == "chat":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = (tokenizer.bos_token or "") + args.prompt
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    if inputs.input_ids.shape[1] + args.max_new_tokens > 512:
        parser.error("Keep prompt plus generation within the trained 512-token window.")
    inputs = inputs.to(args.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.inference_mode():
        output = model.generate(
            inputs=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=50, do_sample=True, use_cache=True,
            eos_token_id=tokenizer.eos_token_id, streamer=streamer,
        )
    continuation = tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
    )
    if args.save_json:
        args.save_json.write_text(json.dumps({
            "mode": args.mode, "prompt": args.prompt, "continuation": continuation,
            "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
            "top_k": 50, "max_new_tokens": args.max_new_tokens,
            "generated_tokens": output.shape[1] - inputs.input_ids.shape[1],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
