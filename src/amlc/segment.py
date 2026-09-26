"""Split concatenated names ('producernavimumbai' -> 'producer navi mumbai') by unigram dynamic programming,
the wordninja idea, but with word frequencies from the split's own S1 names (provided data, no Wikipedia)."""
import math
import re


class Segmenter:
    def __init__(self, counts, min_count=2, max_len=24):
        total = sum(c for c in counts.values() if c >= min_count) or 1
        self.cost = {w: math.log(total / c) for w, c in counts.items() if c >= min_count and w.isalpha()}
        self.max_len = max_len
        self.unknown = math.log(total) + 5.0          # per character: an unknown run is always worse than words

    def _split_alpha(self, s):
        n = len(s)
        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                w = s[j:i]
                c = self.cost.get(w)
                c = c if c is not None else self.unknown * len(w)
                if best[j] + c < best[i]:
                    best[i], back[i] = best[j] + c, j
        words, i = [], n
        while i > 0:
            words.append(s[back[i]:i])
            i = back[i]
        words.reverse()
        # merge adjacent unknown pieces back into one token
        out = []
        for w in words:
            if out and w not in self.cost and out[-1] not in self.cost:
                out[-1] += w
            else:
                out.append(w)
        return out

    def __call__(self, s):
        parts = []
        for chunk in re.findall(r"[a-z]+|\d+", s.lower()):
            parts += self._split_alpha(chunk) if chunk.isalpha() else [chunk]
        return " ".join(parts)
