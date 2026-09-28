import random
import re
from collections import defaultdict, Counter


class BigramModel:
    def __init__(self, corpus, seed=None):
        self.random = random.Random(seed)
        self.bigram_probs = self._build(corpus)

    @staticmethod
    def _tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    def _build(self, corpus):
        counts = defaultdict(Counter)
        for text in corpus:
            words = self._tokenize(text)
            for w1, w2 in zip(words[:-1], words[1:]):
                counts[w1][w2] += 1
        probs = {}
        for w1, nexts in counts.items():
            total = sum(nexts.values())
            probs[w1] = {w2: c / total for w2, c in nexts.items()}
        return probs

    def generate_text(self, start_word, length=10):
        word = start_word.lower()
        out = [word]
        for _ in range(length - 1):
            nexts = self.bigram_probs.get(word)
            if not nexts:
                break
            word = self.random.choices(list(nexts), weights=list(nexts.values()))[0]
            out.append(word)
        return " ".join(out)
