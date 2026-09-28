"""GRPO reward, Eq. (2) of the paper:  r(y; a) = 0.9 * eq(y, a) + 0.1 * fmt(y).

eq(y, a) compares the rollout's answer with the answer the template program COMPUTED from the
parse, so no model judges a rollout. Strings match after lower-casing and stripping punctuation,
articles and plural -s; numbers match numerically within a 5% relative tolerance, like ChartQA's
relaxed accuracy. fmt(y) is 1 when the reply is a short, non-empty answer, as the answer-format
suffix asks (paper A.4.4).

The same eq is used by the blind gate and the difficulty band, so a question is filtered with
exactly the matcher that later pays its rollouts.

ground_truth is the JSON string build_rl_data.py writes: {"answer", "answer_type", "acceptable"}.

Self-test: python vqs/reward/exact_match.py
"""
import json
import re
from typing import Any

REWARD_NAME = "vqs_exact_match"
REWARD_TYPE = "batch"

W_ACC, W_FMT = 0.9, 0.1
NUM_TOL = 0.05
MAX_ANSWER_CHARS = 120

_PUNCT = re.compile(r"[^a-z0-9 .%]+")
_WS = re.compile(r"\s+")
_STOP = {"a", "an", "the", "of", "with"}
_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")
ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def _norm(s):
    return _WS.sub(" ", _PUNCT.sub(" ", str(s).strip().lower())).strip()


def _singular(w):
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def _tokens(s):
    return [_singular(t) for t in _norm(s).split() if t and t not in _STOP]


def _as_num(s):
    m = _NUM.findall(str(s).replace(",", ""))
    return float(m[0]) if m else None


def _value_eq(a, b, tol=NUM_TOL):
    x, y = _as_num(a), _as_num(b)
    if x is None or y is None:
        return False
    return abs(x - y) <= tol * max(abs(y), 1e-9)


def match(pred, gold):
    """eq(y, a) in {0, 1}. `gold` is the dict build_rl_data.gold_of() returns."""
    cands = [str(a) for a in (gold.get("acceptable") or [gold.get("answer", "")])]
    if gold.get("answer_type") == "number":
        return any(_value_eq(pred, c) for c in cands)
    pt = set(_tokens(pred))
    for g in cands:
        if _norm(pred) == _norm(g):
            return True
        gt = set(_tokens(g))
        if gt and pt and (gt <= pt or pt <= gt) and abs(len(pt) - len(gt)) <= 1:
            return True
    return False


def extract_answer(response):
    """The last <answer> tag if the model used one, else the last non-empty line."""
    m = ANS_RE.findall(response)
    if m:
        return m[-1].strip()
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    out = []
    for ri in reward_inputs:
        resp = ri["response"]
        try:
            gold = json.loads(ri["ground_truth"])
        except (TypeError, ValueError):
            gold = {"answer": str(ri["ground_truth"]), "answer_type": "word", "acceptable": []}
        ans = extract_answer(resp)
        acc = 1.0 if ans and match(ans, gold) else 0.0
        fmt = 1.0 if 0 < len(resp.strip()) <= MAX_ANSWER_CHARS else 0.0
        out.append({"overall": W_ACC * acc + W_FMT * fmt, "accuracy": acc, "format": fmt})
    return out


def _selftest():
    num = json.dumps({"answer": "31984", "answer_type": "number", "acceptable": ["31984"]})
    word = json.dumps({"answer": "bench", "answer_type": "word", "acceptable": ["bench"]})
    cases = [("31984", num, 1.0), ("31,984", num, 1.0), ("about 32000", num, 1.0),
             ("40000", num, 0.1), ("", num, 0.0), ("Benches", word, 1.0),
             ("<answer>bench</answer>", word, 1.0), ("sofa", word, 0.1),
             ("bench " * 40, word, 0.9)]
    r = compute_score([{"response": y, "ground_truth": g} for y, g, _ in cases])
    for (y, _, want), s in zip(cases, r):
        print(f"  {y[:30]!r:34s} overall={s['overall']:.2f} acc={s['accuracy']:.0f} "
              f"fmt={s['format']:.0f}")
        assert abs(s["overall"] - want) < 1e-9, f"{y!r}: {s['overall']} != {want}"
    a, b = compute_score([{"response": "bench", "ground_truth": word}] * 2)
    assert a == b, "same input gave different scores"
    print("EXACT_MATCH SELFTEST OK")


if __name__ == "__main__":
    _selftest()
