"""Name normalization / matching and relaxed numeric matching.

Two matching backends for names:
  - lexical (default, deterministic): normalize + token-subset match. Handles plurals and
    modifier-prefixed names ("shoe" ~ "tennis shoe", "man" ~ "young man") without the
    over-matching of raw substring ("ear" in "bear").
  - semantic (opt-in): sentence-transformers cosine >= threshold, lexical as a shortcut.

Also a relaxed numeric parser/matcher (ChartQA convention: |p-g| <= tol*|g|) that copes
with %, $, thousands separators, K/M/B suffixes and parenthesised negatives.
"""
import re

# ---------- name normalization ----------
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_STOP = {"a", "an", "the", "of", "with"}

def _singular(w):
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes") or w.endswith("ches") or w.endswith("shes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def normalize(s):
    s = _PUNCT.sub(" ", str(s).strip().lower())
    s = _WS.sub(" ", s).strip()
    return s

def tokens(s):
    return [_singular(t) for t in normalize(s).split() if t and t not in _STOP]

def norm_key(s):
    """A canonical, order-insensitive key for a name (sorted singularized tokens)."""
    return " ".join(sorted(tokens(s)))

def dedupe(items, key):
    """Unique items by `key`, order-preserving. Used to make tuple sets (SPICE-style)
    so repeated/duplicate entities can neither inflate recall nor tank precision."""
    seen, out = set(), []
    for it in items:
        k = key(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out

def lexical_eq(a, b):
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return normalize(a) == normalize(b)
    # exact token set, or one is a subset of the other (modifier prefix/suffix)
    return ta == tb or ta <= tb or tb <= ta


class Matcher:
    """Name equality with a pluggable backend. Reuse one instance per scoring run."""

    def __init__(self, mode="lexical", embed_model="sentence-transformers/all-MiniLM-L6-v2",
                 threshold=0.6):
        self.mode = mode
        self.threshold = threshold
        self._model = None
        self._canon = None
        self._cache = {}
        if mode == "semantic":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(embed_model)
        elif mode == "gqa":                     # GQA ontology canonicalization (their §3.1)
            from .gqa_canon import GQACanon
            self._canon = GQACanon(threshold=threshold)

    def _emb(self, s):
        key = normalize(s)
        if key not in self._cache:
            self._cache[key] = self._model.encode(key, normalize_embeddings=True)
        return self._cache[key]

    def eq(self, a, b):
        if lexical_eq(a, b):                     # plurals / modifier-prefix robustness
            return True
        if self.mode == "gqa":
            return self._canon.eq(a, b)          # exact match on GQA canonical classes
        if self.mode != "semantic":
            return False
        va, vb = self._emb(a), self._emb(b)
        return float((va * vb).sum()) >= self.threshold


# ---------- greedy set matching ----------
def greedy_match(preds, golds, eq):
    """Greedily match predicted items to gold items (each used at most once).

    Returns (matched_count, matched_pairs). O(len(preds)*len(golds)) — fine for the
    tuple/label counts we see (tens per image).
    """
    used = [False] * len(golds)
    matched = 0
    pairs = []
    for p in preds:
        for j, g in enumerate(golds):
            if not used[j] and eq(p, g):
                used[j] = True
                matched += 1
                pairs.append((p, g))
                break
    return matched, pairs

def prf(matched, n_pred, n_gold):
    """Precision / recall / F1 from a match count and the two set sizes."""
    p = matched / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    r = matched / n_gold if n_gold else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "matched": matched,
            "n_pred": n_pred, "n_gold": n_gold}


# ---------- numeric parsing / relaxed match ----------
_MULT = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12}
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def parse_num(x):
    """Best-effort parse of a scalar to float. Returns None if not numeric.

    Handles: '12,345', '$1.2M', '45%', '(3.1)' (negative), '1.5 billion'.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower().replace(",", "")
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        val = float(m.group().replace(",", ""))
    except ValueError:
        return None
    tail = s[m.end():]
    for suf, mult in _MULT.items():
        if re.match(rf"\s*{suf}\b", tail) or ("billion" in tail and suf == "b") \
                or ("million" in tail and suf == "m") or ("thousand" in tail and suf == "k") \
                or ("trillion" in tail and suf == "t"):
            val *= mult
            break
    if neg:
        val = -abs(val)
    return val

def num_match(pred, gold, tol=0.05):
    """Relaxed numeric match: |pred-gold| <= tol*|gold| (abs tol when gold==0)."""
    gp, gg = parse_num(pred), parse_num(gold)
    if gp is None or gg is None:
        return False
    if gg == 0:
        return abs(gp) <= tol
    return abs(gp - gg) <= tol * abs(gg)

def value_eq(pred, gold, tol=0.05):
    """Match two cell values: numeric-relaxed if both numeric, else normalized string."""
    if parse_num(gold) is not None and parse_num(pred) is not None:
        return num_match(pred, gold, tol)
    return normalize(pred) == normalize(gold) or lexical_eq(pred, gold)
