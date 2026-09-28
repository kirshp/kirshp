"""Обучение GPT на текстовом файле.

    python train.py                         # по умолчанию: data/input.txt, ~15-20 мин на 4-ядерном CPU, ~1 мин на GPU
    python train.py --max_iters 500         # быстрый прогон «проверить, что всё работает»
    python train.py --n_layer 6 --n_embd 384 --block_size 256 --batch_size 64   # на GPU

Результат — out/ckpt.pt (веса + конфиг + словарь).
"""

import argparse
import math
import os
import time

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    """Случайные куски текста: x — вход, y — тот же кусок, сдвинутый на 1 символ вперёд."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, args, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(it, args):
    """Линейный warmup, затем косинусное затухание до 10% от lr."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out_dir", default="out")
    # модель
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    # обучение
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cpu | cuda | mps (по умолчанию — автоопределение)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()

    if not os.path.exists(args.data):
        raise SystemExit(f"Нет файла {args.data}. Запустите: python prepare_data.py (или положите свой текст)")
    with open(args.data, encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}  # 10% текста модель не видит — по нему судим о переобучении
    print(f"Текст: {len(text):,} символов, словарь: {tok.vocab_size} токенов, устройство: {device}")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"Параметров: {model.num_params() / 1e6:.2f} M")

    # AdamW; weight decay применяем только к матрицам (не к bias и LayerNorm)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        for g in optim.param_groups:
            g["lr"] = lr_at(it, args)

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, splits, args, device)
            print(f"шаг {it:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f} | {time.time() - t0:.0f} c")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": cfg.__dict__, "vocab": tok.to_json(), "iter": it},
                    os.path.join(args.out_dir, "ckpt.pt"),
                )
            if it == args.max_iters:
                break

        x, y = get_batch(splits["train"], args.block_size, args.batch_size, device)
        _, loss = model(x, y)                       # 1. прямой проход и ошибка
        optim.zero_grad(set_to_none=True)
        loss.backward()                             # 2. градиенты (backpropagation)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()                                # 3. шаг градиентного спуска

    print(f"Лучший val loss: {best_val:.3f}. Чекпойнт: {args.out_dir}/ckpt.pt")
    print("Генерация: python sample.py")


if __name__ == "__main__":
    main()
