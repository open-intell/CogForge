"""
CogForge — inference.py
========================
Interactive code generation and reasoning with a trained CogForge checkpoint.

Usage:
  python inference.py --checkpoint ./checkpoints/ckpt_step0100000.pt \
                      --prompt "def merge_sort(arr):"
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))

from Architecture import CogForge, CogForgeConfig
from pipline import CodeTokenizer


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    # Rebuild config
    cfg = CogForgeConfig(**{k: v for k, v in cfg_dict.items()
                            if k in CogForgeConfig.__dataclass_fields__})
    model = CogForge(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    tokenizer = CodeTokenizer(vocab_size=cfg.vocab_size)
    return model, tokenizer, cfg


def generate(
    model: CogForge,
    tokenizer: CodeTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    device: str = "cpu",
    show_verifier: bool = False,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    # Only return the newly generated tokens
    generated = out_ids[0, len(ids):].tolist()
    text = tokenizer.decode(generated)

    if show_verifier:
        # Score the full output
        full_out = model(out_ids, return_verifier=True)
        score = full_out["verifier_score"].item()
        print(f"\n[verifier] Predicted correctness score: {score:.3f}")

    return text


def interactive(model: CogForge, tokenizer: CodeTokenizer, device: str):
    print("CogForge Interactive Mode. Type your code prompt (empty line = generate).")
    print("Commands: :quit, :temp <f>, :tokens <n>, :verifier\n")

    temperature = 0.7
    max_tokens = 256
    show_verifier = False

    while True:
        print(">>> ", end="", flush=True)
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.startswith(":quit"):
                return
            if line.startswith(":temp"):
                temperature = float(line.split()[1])
                print(f"Temperature set to {temperature}")
                break
            if line.startswith(":tokens"):
                max_tokens = int(line.split()[1])
                print(f"Max tokens set to {max_tokens}")
                break
            if line.startswith(":verifier"):
                show_verifier = not show_verifier
                print(f"Verifier {'ON' if show_verifier else 'OFF'}")
                break
            if line == "":
                if lines:
                    break
            else:
                lines.append(line)

        if not lines:
            continue

        prompt = "\n".join(lines)
        print("\n--- Generated ---")
        output = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            device=device,
            show_verifier=show_verifier,
        )
        print(output)
        print("-----------------\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verifier", action="store_true")
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    print(f"[inference] Loading checkpoint from {args.checkpoint}")
    model, tokenizer, cfg = load_model(args.checkpoint, args.device)
    print(f"[inference] Model loaded ({model.count_parameters()/1e6:.1f}M params)")

    if args.interactive:
        interactive(model, tokenizer, args.device)
    elif args.prompt:
        out = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=args.device,
            show_verifier=args.verifier,
        )
        print(out)
    else:
        print("Provide --prompt or --interactive")


if __name__ == "__main__":
    main()
