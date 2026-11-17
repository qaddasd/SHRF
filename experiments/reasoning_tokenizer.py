"""Reversible byte-backed vocabulary for the bilingual reasoning experiment."""

import re
from collections import Counter

USER, ASSISTANT, WORK, FINAL, END = range(256, 261)
SPECIAL = ("<user>\n", "\n</user>\n<assistant>\n", "<work>\n", "\n<final>\n", "\n</assistant>\n")
PIECES = re.compile(r"[^\W\d_]+|[0-9]|[ \t]+|\r?\n|.", re.DOTALL)


class ReasoningTokenizer:
    """Use frequent words as tokens, with Unicode character and byte fallback."""

    def __init__(self, words):
        self.words = tuple(words)
        if any(not word or len(word.encode()) < 2 for word in self.words) or len(set(self.words)) != len(self.words):
            raise ValueError("Vocabulary entries must be unique multi-byte strings")
        self.lookup = {word: END+1+i for i, word in enumerate(self.words)}
        self.bytes = ([bytes((value,)) for value in range(256)]
                      + [value.encode() for value in SPECIAL]
                      + [word.encode() for word in self.words])

    @classmethod
    def fit(cls, texts, vocab_size=2048):
        if vocab_size <= END:
            raise ValueError("Vocabulary must fit all bytes and role tokens")
        words, chars = Counter(), Counter()
        for text in texts:
            words.update(piece for piece in PIECES.findall(text) if 2 <= len(piece.encode()) <= 36)
            chars.update(char for char in text if ord(char) > 127)
        capacity = vocab_size-END-1
        fallback = [char for char, _ in chars.most_common(min(capacity, 96))]
        for char in fallback:
            words.pop(char, None)
        ranked = sorted(words, key=lambda word: (-(len(word.encode())-1)*words[word], word))
        return cls(fallback + ranked[:capacity-len(fallback)])

    def encode(self, text):
        tokens = []
        for piece in PIECES.findall(text):
            if piece in self.lookup:
                tokens.append(self.lookup[piece])
                continue
            for char in piece:
                if char in self.lookup:
                    tokens.append(self.lookup[char])
                else:
                    tokens.extend(char.encode())
        return tokens

    def decode(self, tokens):
        return b"".join(self.bytes[token] for token in tokens).decode("utf-8", errors="replace")

    def prompt(self, text):
        return [USER, *self.encode(text), ASSISTANT]
