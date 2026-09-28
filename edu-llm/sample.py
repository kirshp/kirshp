"""Генерация текста обученной моделью.

    python sample.py
    python sample.py --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top_k 20
"""

import argparse

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8, help="<1 — осторожнее, >1 — «креативнее»")
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = args.device or pick_device()

    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.from_json(ckpt["vocab"])

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
