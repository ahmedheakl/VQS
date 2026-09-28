"""Self-evolve reward: answer exact-match + EVIDENCE F1 + format.

Why the evidence term exists. Our measurements say 80-88% of questions are answered 0/k or
k/k by the policy, and GRPO gets zero advantage from a group with no spread -- that is the
whole reason the executable-reward arm landed at -1.4. Evidence F1 is dense: on a question the
policy never answers right, rollouts still differ in WHAT they read off the image, so the group
has spread and therefore gradient.

Anti-hack: evidence recall alone is farmed by dumping the whole chart into <see> (the base model
already does this -- it listed 14 bars for a 2-value question). So we score F1, not recall, and
hard-cap the block at MAX_EV_LINES lines before scoring.

ground_truth is a JSON string:
  {"answer", "answer_type", "acceptable", "evidence": [[label, value], ...], "scaffold": bool}
`scaffold` marks the rollouts that were prompted for a <see> block. Plain rollouts are scored on
the answer alone, so the policy is optimised for the exact format used at eval time.

Self-test: python examples/reward_function/se_evidence.py
"""
import json
import re
from typing import Any

REWARD_NAME = "se_evidence"
REWARD_TYPE = "batch"

MAX_EV_LINES = 4
W_SCAFFOLD = {"acc": 0.70, "ev": 0.20, "fmt": 0.10}
W_PLAIN = {"acc": 0.90, "ev": 0.00, "fmt": 0.10}

_PUNCT = re.compile(r"[^a-z0-9 .%]+")
_WS = re.compile(r"\s+")
_STOP = {"a", "an", "the", "of", "with"}
_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")
SEE_RE = re.compile(r"<see>(.*?)</see>", re.S | re.I)
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


def _value_eq(a, b, tol=0.05):
    x, y = _as_num(a), _as_num(b)
    if x is None or y is None:
        return False
    return abs(x - y) <= tol * max(abs(y), 1e-9)


def _match(pred, gold):
    """Answer match. Numbers use 5% relaxed tolerance, like ChartQA's own metric."""
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


def parse_evidence(response):
    """Lines inside <see>, capped. The cap is the anti-dumping guard, applied BEFORE scoring."""
    m = SEE_RE.search(response)
    if not m:
        return []
    out = []
    for line in m.group(1).strip().splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        k, _, v = line.partition(":")
        out.append((k.strip(), v.strip()) if v.strip() else (line, line))
        if len(out) == MAX_EV_LINES:
            break
    return out


def evidence_f1(pred_pairs, gold_pairs):
    """A gold pair is recovered if some predicted line matches its label AND its value."""
    if not gold_pairs:
        return 0.0
    if not pred_pairs:
        return 0.0
    used, hits = set(), 0
    for gl, gv in gold_pairs:
        for i, (pl, pv) in enumerate(pred_pairs):
            if i in used:
                continue
            lab_ok = _norm(pl) == _norm(gl) or set(_tokens(gl)) <= set(_tokens(pl)) \
                or set(_tokens(pl)) <= set(_tokens(gl))
            val_ok = _norm(pv) == _norm(gv) or _value_eq(pv, gv)
            if lab_ok and val_ok and set(_tokens(gl)):
                used.add(i)
                hits += 1
                break
    prec, rec = hits / len(pred_pairs), hits / len(gold_pairs)
    return 0.0 if hits == 0 else 2 * prec * rec / (prec + rec)


def extract_answer(response, scaffold):
    m = ANS_RE.findall(response)
    if m:
        return m[-1].strip()
    if scaffold:
        return ""          # scaffold rollouts must use the tag; no free pass
    body = SEE_RE.sub("", response).strip()
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    out = []
    for ri in reward_inputs:
        resp = ri["response"]
        try:
            gold = json.loads(ri["ground_truth"])
        except (TypeError, ValueError):
            gold = {"answer": str(ri["ground_truth"]), "answer_type": "word",
                    "acceptable": [], "evidence": [], "scaffold": False}
        scaffold = bool(gold.get("scaffold"))
        w = W_SCAFFOLD if scaffold else W_PLAIN

        ans = extract_answer(resp, scaffold)
        acc = 1.0 if ans and _match(ans, gold) else 0.0
        if scaffold:
            fmt = 1.0 if (SEE_RE.search(resp) and ANS_RE.search(resp)) else 0.0
            ev = evidence_f1(parse_evidence(resp), gold.get("evidence") or [])
        else:
            fmt = 1.0 if 0 < len(resp.strip()) <= 120 else 0.0
            ev = 0.0
        out.append({"overall": w["acc"] * acc + w["ev"] * ev + w["fmt"] * fmt,
                    "accuracy": acc, "evidence": ev, "format": fmt,
                    "scaffold": 1.0 if scaffold else 0.0})
    return out


def _selftest():
    gold = {"answer": "6", "answer_type": "number", "acceptable": ["6"],
            "evidence": [["Mexico", "1.7"], ["Argentina", "0.2"]], "scaffold": True}
    gj = json.dumps(gold)
    perfect = "<see>\nMexico: 1.7\nArgentina: 0.2\n</see>\n<answer>6</answer>"
    noev = "<see>\n</see>\n<answer>6</answer>"
    wrong_ans = "<see>\nMexico: 1.7\nArgentina: 0.2\n</see>\n<answer>99</answer>"
    dump = "<see>\n" + "\n".join(f"C{i}: {i}" for i in range(14)) + "\nMexico: 1.7\n</see><answer>6</answer>"
    junk = "<see>\nfoo: bar\nbaz: qux\n</see>\n<answer>6</answer>"
    r = compute_score([{"response": x, "ground_truth": gj}
                       for x in (perfect, noev, wrong_ans, dump, junk)])
    names = ["perfect", "correct-no-evidence", "WRONG-answer-good-evidence", "dump-14-lines", "junk-evidence"]
    for n, s in zip(names, r):
        print(f"  {n:28s} overall={s['overall']:.3f} acc={s['accuracy']:.0f} "
              f"ev={s['evidence']:.3f} fmt={s['format']:.0f}")
    assert abs(r[0]["overall"] - 1.0) < 1e-9, "perfect response must score 1.0"
    assert r[1]["evidence"] == 0.0 and r[1]["accuracy"] == 1.0
    # THE point of the design: a wrong answer still carries signal
    assert r[2]["accuracy"] == 0.0 and r[2]["evidence"] == 1.0, "dense gradient on dead questions"
    assert r[2]["overall"] > r[1]["overall"] - 0.75, "evidence must move the score"
    assert r[3]["evidence"] < 0.35, f"dumping must be punished, got {r[3]['evidence']}"
    assert r[4]["evidence"] == 0.0, "junk evidence must score 0"

    # plain rows ignore evidence entirely
    gp = json.dumps({**gold, "scaffold": False})
    p = compute_score([{"response": "6", "ground_truth": gp}])[0]
    assert p["accuracy"] == 1.0 and p["evidence"] == 0.0 and abs(p["overall"] - 1.0) < 1e-9, p
    # zero-strength control: identical responses must give identical scores
    a, b = compute_score([{"response": perfect, "ground_truth": gj}] * 2)
    assert a == b, "same input gave different scores"
    print("SE_EVIDENCE SELFTEST OK")


if __name__ == "__main__":
    _selftest()
