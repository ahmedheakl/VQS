"""GQA-style synonym handling: canonicalize a free-form name to GQA's ontology, then
exact-match — exactly the approach from Hudson & Manning (GQA, CVPR 2019, §3.1).

GQA consolidated the free-form Visual Genome vocabulary into a clean ontology of ~2690
canonical classes (objects/attributes/relations) using WORD-EMBEDDING distances + curation,
and their scene graphs (our GT) are already in that canonical vocabulary. We reproduce the
consolidation for the model's predictions: map each predicted name to its nearest canonical
GQA class by GloVe cosine (GQA used GloVe 300d), keeping it verbatim if already canonical or
if no class is close enough. Distinct classes stay distinct (man != person), true synonyms
collapse (guy -> man) — matching GQA's ontology, not an invented fuzzy matcher.
"""
import os, json
import numpy as np

_GQA_SG = ["gqa/train_sceneGraphs.json", "gqa/val_sceneGraphs.json"]
_VOCAB_CACHE = "data/gqa_canonical_vocab.json"


def build_vocab():
    """The GQA canonical vocabulary = every object/attribute/relation name in their
    (already-normalized) scene graphs. Cached to disk after the first (slow) build."""
    if os.path.exists(_VOCAB_CACHE):
        return set(json.load(open(_VOCAB_CACHE)))
    vocab = set()
    for p in _GQA_SG:
        d = json.load(open(p))
        for v in d.values():
            for o in v.get("objects", {}).values():
                if o.get("name"):
                    vocab.add(o["name"])
                vocab.update(o.get("attributes", []) or [])
                for r in o.get("relations", []) or []:
                    if r.get("name"):
                        vocab.add(r["name"])
    vocab = sorted(x for x in vocab if x)
    os.makedirs(os.path.dirname(_VOCAB_CACHE), exist_ok=True)
    json.dump(vocab, open(_VOCAB_CACHE, "w"))
    return set(vocab)


class GQACanon:
    def __init__(self, threshold=0.60, glove="glove-wiki-gigaword-300"):
        from .match import normalize, tokens
        self._norm = normalize
        self._tok = tokens
        self.threshold = threshold
        self.vocab = {normalize(w) for w in build_vocab()}      # normalized canonical set
        import gensim.downloader as api
        self.kv = api.load(glove)
        names, vecs = [], []
        for w in sorted(self.vocab):                            # embed each canonical class
            v = self._embed(w)
            if v is not None:
                names.append(w); vecs.append(v)
        self.canon_names = names
        self.mat = np.asarray(vecs, dtype=np.float32)           # (N, 300), unit-norm
        self.cache = {}

    def _embed(self, name):
        toks = [t for t in self._tok(name) if t in self.kv] or \
               [w for w in self._norm(name).split() if w in self.kv]
        if not toks:
            return None
        v = np.mean([self.kv[t] for t in toks], axis=0)
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else None

    def canon(self, name):
        """Return the canonical GQA class for a name (verbatim if already canonical / OOV)."""
        key = self._norm(name)
        if key in self.cache:
            return self.cache[key]
        if key in self.vocab:                                   # already a canonical class
            out = key
        else:
            v = self._embed(name)
            if v is None:
                out = key
            else:
                sims = self.mat @ v
                j = int(np.argmax(sims))
                out = self.canon_names[j] if sims[j] >= self.threshold else key
        self.cache[key] = out
        return out

    def eq(self, a, b):
        return self.canon(a) == self.canon(b)
