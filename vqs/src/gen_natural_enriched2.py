"""Second batch of enriched NATURAL families (the rest of the audit taxonomy), over GOLD graphs.
Answer-blind, correct-by-construction; every family has a recompute() branch so --selftest can prove
100% reproducibility. OCR / scene_recognition / extended_attr are intentionally omitted (need pixels).

Families:
  state_attr (L2,choice) action_attr (L2,T/F) object_recognition (L2,wh) gated_object_recognition (L3,wh)
  state_filtered_existence (L3,T/F) action_filtered_existence (L3,T/F) filtered_frame_position (L3,choice)
  state_gated_relation (L3,choice) action_gated_relation (L3,T/F) frame_gated_relation (L4,T/F)
  relation_gated_action (L3,choice) relation_chain (L4,T/F) group_attribute (L3,T/F)
  or_filtered_existence (L4,T/F) multi_gated_relation (L4,T/F) gated_attribute_conjunction (L4,T/F)
"""
import random
from . import vqa_engine as ve
from . import gen_natural_enriched as ge
from .vqa_engine import _natural_index, _good_referent, _typed_attr, _norm, _is_plural, ATTR_TYPES

STATE = {"dark", "clear", "open", "parked", "cloudy", "empty", "closed", "bare", "light", "bright",
         "wet", "dirty", "dry", "calm", "clean", "snowy", "covered", "cooked", "sliced", "painted",
         "tiled", "paved", "framed", "filled", "full", "hanging"}
POSE = {"standing", "sitting", "walking", "running", "jumping", "lying", "flying", "grazing",
        "eating", "surfing", "playing", "swimming", "waiting", "smiling", "kneeling", "bending",
        "crouching", "leaning", "laying", "skiing", "sleeping", "resting"}
ACTION_VERBS = {"wearing", "holding", "carrying", "riding", "watching", "eating", "using",
                "looking at", "covering", "crossing", "reading", "drinking", "catching", "throwing",
                "pushing", "pulling", "feeding", "petting", "driving", "hitting", "kicking",
                "swinging", "playing with", "riding on"}
OPP = {"open": "closed", "closed": "open", "clean": "dirty", "dirty": "clean", "wet": "dry",
       "dry": "wet", "light": "dark", "dark": "light", "empty": "full", "full": "empty",
       "bright": "dark"}
HYPER = {
    "furniture": {"chair", "table", "desk", "sofa", "couch", "bed", "cabinet", "shelf", "bench",
                  "stool", "drawer", "dresser", "nightstand", "wardrobe", "bookshelf", "ottoman"},
    "animal": {"dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "giraffe", "zebra", "bear",
               "duck", "goat", "pig", "lion", "monkey", "deer", "cattle", "donkey"},
    "vehicle": {"car", "truck", "bus", "motorcycle", "bike", "bicycle", "train", "boat", "van",
                "jeep", "scooter", "cab", "taxi"},
    "aircraft": {"airplane", "plane", "jet", "helicopter", "aircraft"},
    "fruit": {"banana", "apple", "orange", "strawberry", "grape", "pear", "lemon", "peach"},
}
DIRS = [("to the left of", "left"), ("to the right of", "right"), ("above", "above"), ("below", "below")]
PHRASE2KEY = {p: k for p, k in DIRS}


def _attrs(o):
    return {_norm(a) for a in (o.get("attributes", []) or [])}


def _side(o, W):
    c = ge._center(o)
    if not c:
        return None
    return "left" if c[0] < W * 0.44 else ("right" if c[0] > W * 0.56 else None)


def _in_center(o, W, H):
    c = ge._center(o)
    return bool(c) and W * 0.33 < c[0] < W * 0.66 and H * 0.33 < c[1] < H * 0.66


def _action_rels(o, id2name):
    out = []
    for r in (o.get("relations", []) or []):
        p = _norm(r.get("name", ""))
        t = _norm(id2name.get(r.get("object", ""), ""))
        if p in ACTION_VERBS and t:
            out.append((p, t))
    return out


def gen(entry, rng, vocab, per_family=1):
    objs, id2name, name_count = _natural_index(entry)
    W, H = entry.get("width") or 1, entry.get("height") or 1
    pres = ge._present(entry)
    present = set(pres)
    uniq = [n for n in present if name_count.get(n) == 1 and not _is_plural(n)
            and _good_referent(pres[n][0][1], n, W, H)]
    out = []

    def add(fam, diff, q, ans, fmt, slots):
        p = {"domain": "natural", "family": fam, "difficulty": diff, "question": q,
             "answer": str(ans).lower(), "answer_type": "word", "format": fmt}
        p.update(slots)
        out.append(p)

    # state_attr (choice over an opposite pair the object actually resolves)
    sa = []
    for n in uniq:
        a = _attrs(pres[n][0][1])
        for s in a & set(OPP):
            if OPP[s] not in a:
                sa.append((n, s))
                break
    rng.shuffle(sa)
    for n, s in sa[:per_family]:
        opts = sorted([s, OPP[s]])
        add("state_attr", 2, f"Is the {n} {opts[0]} or {opts[1]}?", s, "C", {"subj": n})

    # action_attr (yes/no over a pose attribute)
    aa = []
    for n in uniq:
        a = _attrs(pres[n][0][1])
        p = a & POSE
        if p:
            aa.append((n, next(iter(p)), "yes"))
        else:
            aa.append((n, rng.choice(sorted(POSE)), "no"))
    rng.shuffle(aa)
    for n, pose, ans in aa[:per_family * 2]:
        add("action_attr", 2, f"Is the {n} {pose}?", ans, "T" if ans == "yes" else "F",
            {"subj": n, "pose": pose})

    # object_recognition / gated_object_recognition (exactly ONE category KIND present -> well-posed)
    for cat, members in HYPER.items():
        kinds = {n for n in present if n in members}
        if len(kinds) == 1:
            add("object_recognition", 2, f"What kind of {cat} is in the picture?", next(iter(kinds)),
                "W", {"cat": cat})
        for atype in ATTR_TYPES:
            val2kinds = {}
            for n in (x for x in present if x in members):
                for _, o in pres[n]:
                    v = _typed_attr(o.get("attributes", []), atype)
                    if v:
                        val2kinds.setdefault(_norm(v), set()).add(n)
            for v, ks in val2kinds.items():
                if len(ks) == 1:
                    add("gated_object_recognition", 3, f"Which kind of {cat} is {v}?",
                        next(iter(ks)), "W", {"cat": cat, "gate_attr": v, "atype": atype})
                    break

    # state_filtered_existence / action_filtered_existence (yes/no)
    for n in uniq[:per_family * 3]:
        a = _attrs(pres[n][0][1])
        st = a & STATE
        if st:
            add("state_filtered_existence", 3, f"Do you see a {next(iter(st))} {n} in the picture?",
                "yes", "T", {"q_name": n, "q_state": next(iter(st))})
        else:
            add("state_filtered_existence", 3,
                f"Do you see a {rng.choice(sorted(STATE))} {n} in the picture?", "no", "F",
                {"q_name": n, "q_state": "__none__"})
        ars = _action_rels(pres[n][0][1], id2name)
        pz = a & POSE
        if ars:
            add("action_filtered_existence", 3, f"Is there a {n} that is {ars[0][0]} something?",
                "yes", "T", {"q_name": n, "q_action": ars[0][0], "kind": "rel"})
        elif pz:
            add("action_filtered_existence", 3, f"Is there a {n} that is {next(iter(pz))}?", "yes",
                "T", {"q_name": n, "q_action": next(iter(pz)), "kind": "pose"})

    # filtered_frame_position (subject picked by a typed attribute, side from box)
    ffp = []
    for n in uniq:
        o = pres[n][0][1]
        side = _side(o, W)
        if not side:
            continue
        for atype in ATTR_TYPES:
            v = _typed_attr(o.get("attributes", []), atype)
            if v:
                ffp.append((_norm(v), n, side)); break
    rng.shuffle(ffp)
    for v, n, side in ffp[:per_family]:
        add("filtered_frame_position", 3,
            f"Is the {v} {n} on the left or the right side of the picture?", side, "C",
            {"subj": n, "gate_attr": v})

    # group_attribute (two named objects share a typed attribute value?)
    ga = []
    for atype in ATTR_TYPES:
        withv = [(n, _norm(_typed_attr(pres[n][0][1].get("attributes", []), atype))) for n in uniq
                 if _typed_attr(pres[n][0][1].get("attributes", []), atype)]
        if len(withv) >= 2:
            (a, va), (b, vb) = rng.sample(withv, 2)
            if va == vb:
                add("group_attribute", 3, f"Do the {a} and the {b} both have {va} color?"
                    if atype == "color" else f"Are the {a} and the {b} both {va}?", "yes", "T",
                    {"a": a, "b": b, "val": va})
            else:
                add("group_attribute", 3, f"Are the {a} and the {b} both {va}?", "no", "F",
                    {"a": a, "b": b, "val": va})

    # or_filtered_existence (L4): (A or B) with a shared attribute
    cls_with = {}
    for n in present:
        vals = set()
        for _, o in pres[n]:
            for atype in ATTR_TYPES:
                v = _typed_attr(o.get("attributes", []), atype)
                if v:
                    vals.add(_norm(v))
        if vals:
            cls_with[n] = vals
    keys = list(cls_with)
    if len(keys) >= 2:
        a, b = rng.sample(keys, 2)
        v = rng.choice(sorted(cls_with[a] | cls_with[b]))
        truth = (v in cls_with[a]) or (v in cls_with[b])
        add("or_filtered_existence", 4, f"Are there any {ge._plural(a)} or {ge._plural(b)} that are {v}?",
            "yes" if truth else "no", "T" if truth else "F", {"a": a, "b": b, "val": v})

    # ---- box-relative spatial families ----
    def dir_between(sub, anc):
        sc, bc = ge._center(sub), ge._center(anc)
        if not sc or not bc:
            return None
        return "left" if sc[0] < bc[0] else "right"

    # state_gated_relation (choice: which side of anchor is the state-gated subject)
    for _ in range(per_family):
        anc = rng.choice(uniq) if uniq else None
        cands = [(n, s) for n in uniq for s in (_attrs(pres[n][0][1]) & STATE) if n != anc]
        if anc and cands:
            n, s = rng.choice(cands)
            d = dir_between(pres[n][0][1], pres[anc][0][1])
            if d:
                add("state_gated_relation", 3,
                    f"Are the {s} {n} to the left or to the right of the {anc}?", d, "C",
                    {"subj": n, "anchor": anc})
                break

    # action_gated_relation (yes/no: subj to dir of an anchor that performs an action)
    for anc in uniq:
        ars = _action_rels(pres[anc][0][1], id2name)
        if not ars:
            continue
        pred, tgt = ars[0]
        subj = next((n for n in present if n != anc and n != tgt and ge._center(pres[n][0][1])), None)
        if not subj:
            continue
        bc = ge._center(pres[anc][0][1])
        phrase, key = rng.choice(DIRS)
        truth = any(ge._dir_true(key, ge._center(o), bc, W, H) for _, o in pres[subj] if ge._center(o))
        add("action_gated_relation", 3,
            f"Are there any {ge._plural(subj)} {phrase} the {anc} that is {pred} the {tgt}?",
            "yes" if truth else "no", "T" if truth else "F",
            {"subj": subj, "anchor": anc, "dir": phrase, "pred": pred, "tgt": tgt})
        break

    # frame_gated_relation (yes/no: subj to dir of an anchor in the center of the frame)
    center_anchors = [n for n in uniq if _in_center(pres[n][0][1], W, H)]
    if center_anchors:
        anc = center_anchors[0]
        bc = ge._center(pres[anc][0][1])
        subj = next((n for n in present if n != anc and ge._center(pres[n][0][1])), None)
        if subj:
            phrase, key = rng.choice(DIRS)
            truth = any(ge._dir_true(key, ge._center(o), bc, W, H) for _, o in pres[subj] if ge._center(o))
            add("frame_gated_relation", 4,
                f"Are there any {ge._plural(subj)} {phrase} the {anc} in the center of the image?",
                "yes" if truth else "no", "T" if truth else "F",
                {"subj": subj, "anchor": anc, "dir": phrase})

    # relation_gated_action (choice: pose of the unique object on one side of an anchor)
    for anc in uniq:
        bc = ge._center(pres[anc][0][1])
        for phrase, key in DIRS:
            found = [(n, pres[n][0][1]) for n in uniq if n != anc
                     and ge._center(pres[n][0][1]) and ge._dir_true(key, ge._center(pres[n][0][1]), bc, W, H)]
            posed = [(n, (_attrs(o) & POSE)) for n, o in found if (_attrs(o) & POSE)]
            if len(posed) == 1 and len(posed[0][1]) == 1:
                n, poses = posed[0]
                pose = next(iter(poses))
                alt_pool = [p for p in sorted(POSE) if p not in poses]
                if not alt_pool:
                    break
                alt = rng.choice(alt_pool)
                opts = sorted([pose, alt])
                add("relation_gated_action", 3,
                    f"Is the {n} {phrase} the {anc} {opts[0]} or {opts[1]}?", pose, "C",
                    {"subj": n, "anchor": anc, "dir": phrase, "opts": opts})
                break
        else:
            continue
        break

    # relation_chain (yes/no, 2-hop spatial): A dir1 the B that is dir2 the C
    for c in uniq:                       # anchor C
        cc = ge._center(pres[c][0][1])
        if not cc:
            continue
        for phrase2, key2 in DIRS:
            mids = [b for b in uniq if b != c and ge._center(pres[b][0][1])
                    and ge._dir_true(key2, ge._center(pres[b][0][1]), cc, W, H)]
            if len(mids) != 1:
                continue
            b = mids[0]
            bc = ge._center(pres[b][0][1])
            a = next((n for n in present if n not in (b, c) and ge._center(pres[n][0][1])), None)
            if not a:
                continue
            phrase1, key1 = rng.choice(DIRS)
            truth = any(ge._dir_true(key1, ge._center(o), bc, W, H) for _, o in pres[a] if ge._center(o))
            add("relation_chain", 4,
                f"Are there any {ge._plural(a)} {phrase1} the {b} that is {phrase2} the {c}?",
                "yes" if truth else "no", "T" if truth else "F",
                {"a": a, "b": b, "c": c, "dir1": phrase1, "dir2": phrase2})
            break
        else:
            continue
        break

    # multi_gated_relation (L4): the {attr}{A} dir the {B} that is {action} the {C}
    for anc in uniq:
        ars = _action_rels(pres[anc][0][1], id2name)
        if not ars:
            continue
        pred, tgt = ars[0]
        subj = None
        for n in uniq:
            if n in (anc, tgt):
                continue
            v = None
            for atype in ATTR_TYPES:
                v = _typed_attr(pres[n][0][1].get("attributes", []), atype)
                if v:
                    break
            if v and ge._center(pres[n][0][1]):
                subj, sv = n, _norm(v)
                break
        if not subj:
            continue
        bc = ge._center(pres[anc][0][1])
        phrase, key = rng.choice(DIRS)
        truth = ge._dir_true(key, ge._center(pres[subj][0][1]), bc, W, H)
        add("multi_gated_relation", 4,
            f"Is the {sv} {subj} {phrase} the {anc} that is {pred} the {tgt}?",
            "yes" if truth else "no", "T" if truth else "F",
            {"subj": subj, "gate_attr": sv, "anchor": anc, "dir": phrase, "pred": pred, "tgt": tgt})
        break

    # gated_attribute_conjunction (L4): the {A} {rel} the {B} look {a1} and {a2}
    for a in uniq:
        rels = [(_norm(r.get("name", "")), _norm(id2name.get(r.get("object", ""), "")))
                for r in (pres[a][0][1].get("relations", []) or [])]
        rels = [(p, t) for p, t in rels if t and name_count.get(t) == 1 and p in
                (ACTION_VERBS | {"behind", "in front of", "near", "next to", "under", "on"})]
        if not rels:
            continue
        pred, tgt = rels[0]
        to = pres[tgt][0][1]
        typed = [_norm(_typed_attr(to.get("attributes", []), at)) for at in ATTR_TYPES
                 if _typed_attr(to.get("attributes", []), at)]
        if len(typed) >= 2:
            add("gated_attribute_conjunction", 4,
                f"Does the {tgt} that the {a} is {pred} look {typed[0]} and {typed[1]}?", "yes", "T",
                {"anchor": a, "pred": pred, "tgt": tgt, "a1": typed[0], "a2": typed[1]})
            break
    return out


# ------------------------------------------------------------------ independent recompute
def recompute(prog, entry):
    fam = prog["family"]
    objs, id2name, name_count = _natural_index(entry)
    W, H = entry.get("width") or 1, entry.get("height") or 1
    pres = ge._present(entry)
    present = set(pres)

    def uniq_obj(n):
        return pres[n][0][1] if name_count.get(n) == 1 and n in pres else None

    if fam == "state_attr":
        o = uniq_obj(prog["subj"])
        if not o:
            return None
        a = _attrs(o)
        got = [s for s in a & set(OPP) if OPP[s] not in a]
        return got[0] if len(got) == 1 else (got[0] if got else None)
    if fam == "action_attr":
        o = uniq_obj(prog["subj"])
        return None if not o else ("yes" if prog["pose"] in _attrs(o) else "no")
    if fam == "object_recognition":
        kinds = {n for n in present if n in HYPER[prog["cat"]]}
        return next(iter(kinds)) if len(kinds) == 1 else None
    if fam == "gated_object_recognition":
        cat, v, at = prog["cat"], prog["gate_attr"], prog["atype"]
        kinds = set()
        for n in (x for x in present if x in HYPER[cat]):
            for _, o in pres[n]:
                if _norm(_typed_attr(o.get("attributes", []), at) or "") == v:
                    kinds.add(n)
        return next(iter(kinds)) if len(kinds) == 1 else None
    if fam == "state_filtered_existence":
        n = prog["q_name"]
        if prog["q_state"] == "__none__":
            return "no"
        return "yes" if any(prog["q_state"] in _attrs(o) for _, o in pres.get(n, [])) else "no"
    if fam == "action_filtered_existence":
        n, act, kind = prog["q_name"], prog["q_action"], prog["kind"]
        for _, o in pres.get(n, []):
            if kind == "pose" and act in _attrs(o):
                return "yes"
            if kind == "rel" and any(_norm(r.get("name", "")) == act for r in (o.get("relations", []) or [])):
                return "yes"
        return "no"
    if fam == "filtered_frame_position":
        o = uniq_obj(prog["subj"])
        return _side(o, W) if o else None
    if fam == "group_attribute":
        a, b, val = prog["a"], prog["b"], prog["val"]
        oa, ob = uniq_obj(a), uniq_obj(b)
        if not oa or not ob:
            return None
        return "yes" if (val in _attrs(oa) and val in _attrs(ob)) else "no"
    if fam == "or_filtered_existence":
        v = prog["val"]
        def has(n):
            return any(v == _norm(_typed_attr(o.get("attributes", []), at) or "")
                       for _, o in pres.get(n, []) for at in ATTR_TYPES)
        return "yes" if (has(prog["a"]) or has(prog["b"])) else "no"
    if fam in ("state_gated_relation",):
        n, anc = prog["subj"], prog["anchor"]
        os_, oa = uniq_obj(n), uniq_obj(anc)
        if not os_ or not oa:
            return None
        sc, ac = ge._center(os_), ge._center(oa)
        return None if not sc or not ac else ("left" if sc[0] < ac[0] else "right")
    if fam in ("action_gated_relation", "frame_gated_relation"):
        subj, anc = prog["subj"], prog["anchor"]
        oa = uniq_obj(anc)
        if not oa or subj not in pres:
            return None
        bc = ge._center(oa)
        key = PHRASE2KEY[prog["dir"]]
        return "yes" if any(ge._dir_true(key, ge._center(o), bc, W, H) for _, o in pres[subj] if ge._center(o)) else "no"
    if fam == "relation_gated_action":
        o = uniq_obj(prog["subj"])
        if not o:
            return None
        got = _attrs(o) & set(prog["opts"])
        return next(iter(got)) if len(got) == 1 else None
    if fam == "relation_chain":
        a, b, c = prog["a"], prog["b"], prog["c"]
        ob, oc = uniq_obj(b), uniq_obj(c)
        if not ob or not oc or a not in pres:
            return None
        if not ge._dir_true(PHRASE2KEY[prog["dir2"]], ge._center(ob), ge._center(oc), W, H):
            return None
        bc = ge._center(ob)
        return "yes" if any(ge._dir_true(PHRASE2KEY[prog["dir1"]], ge._center(o), bc, W, H)
                            for _, o in pres[a] if ge._center(o)) else "no"
    if fam == "multi_gated_relation":
        subj, anc = prog["subj"], prog["anchor"]
        os_, oa = uniq_obj(subj), uniq_obj(anc)
        if not os_ or not oa:
            return None
        return "yes" if ge._dir_true(PHRASE2KEY[prog["dir"]], ge._center(os_), ge._center(oa), W, H) else "no"
    if fam == "gated_attribute_conjunction":
        tgt = uniq_obj(prog["tgt"])
        if not tgt:
            return None
        a = _attrs(tgt)
        return "yes" if (prog["a1"] in a and prog["a2"] in a) else "no"
    return None


FAMILIES = {"state_attr", "action_attr", "object_recognition", "gated_object_recognition",
            "state_filtered_existence", "action_filtered_existence", "filtered_frame_position",
            "state_gated_relation", "action_gated_relation", "frame_gated_relation",
            "relation_gated_action", "relation_chain", "group_attribute", "or_filtered_existence",
            "multi_gated_relation", "gated_attribute_conjunction"}
