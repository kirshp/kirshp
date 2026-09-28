"""Минимальный GPT (decoder-only трансформер) для обучения.

Весь «интеллект» модели — это функция, которая по последовательности токенов
предсказывает распределение вероятностей следующего токена. Ниже она собрана
из четырёх деталей:

    1. Embedding       — токен и позиция -> вектор размерности n_embd
    2. CausalSelfAttention — токены «смотрят» на предыдущие токены
    3. MLP             — поточечная нелинейная обработка каждого вектора
    4. LM head         — вектор -> логиты по словарю (что будет дальше)

Блоки 2+3 с residual-связями и LayerNorm повторяются n_layer раз.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65      # размер словаря (для посимвольной токенизации ~65-150)
    block_size: int = 128     # максимальная длина контекста в токенах
    n_layer: int = 4          # число трансформер-блоков
    n_head: int = 4           # число голов внимания
    n_embd: int = 128         # размерность скрытого вектора (d_model)
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention с каузальной маской.

    Для каждого токена считаем три вектора:
        q (query) — «что я ищу»,
        k (key)   — «что я содержу»,
        v (value) — «что я отдам, если меня выберут».
    Вес внимания токена i к токену j = softmax_j(q_i · k_j / sqrt(d)).
    Маска запрещает смотреть в будущее (j > i) — иначе модель «подглядит» ответ.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # q, k, v одним матричным умножением
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # смешиваем выходы голов
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Нижнетреугольная матрица: 1 — можно смотреть, 0 — нельзя
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, время (длина), каналы (n_embd)
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): каждая голова работает независимо
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                                 # (B, h, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, C)            # склеиваем головы
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Feed-forward сеть: расширяем в 4 раза, GELU, сжимаем обратно."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Трансформер-блок (pre-norm): x + Attn(LN(x)), затем x + MLP(LN(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # общение между токенами
        x = x + self.mlp(self.ln2(x))   # «размышление» каждого токена в отдельности
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # обучаемые позиции
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: одна и та же матрица кодирует и декодирует токены
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx: (B, T) индексы токенов. targets: (B, T) — те же токены, сдвинутые на 1."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"контекст {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Кросс-энтропия = -log p(правильный следующий токен), усреднённая по всем позициям
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Авторегрессия: предсказали токен -> дописали -> повторили."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]          # обрезаем до окна контекста
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # берём только последнюю позицию
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")    # отсекаем маловероятные
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # случайный выбор по вероятностям
            idx = torch.cat([idx, next_id], dim=1)
        return idx
