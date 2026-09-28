"""Посимвольный токенизатор: каждый уникальный символ текста = один токен.

Самый простой и наглядный вариант. Настоящие LLM используют BPE (подслова),
чтобы последовательности были короче, но принцип тот же: текст <-> список чисел.
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.unk = self.stoi.get(" ", 0)  # незнакомые символы при генерации заменяем пробелом

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi.get(ch, self.unk) for ch in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def to_json(self):
        return json.dumps(self.chars, ensure_ascii=False)

    @classmethod
    def from_json(cls, s):
        return cls(json.loads(s))
