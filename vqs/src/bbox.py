"""Bounding-box helpers: normalization, parsing, IoU. All boxes are [x, y, w, h]
(top-left + size). Targets/predictions use coords normalized to [0,1] (resolution-free)."""

def norm_box(x, y, w, h, W, H):
    """Pixel box -> [x,y,w,h] normalized to [0,1] and clamped. None if image size unknown."""
    if not W or not H:
        return None
    def cl(v):
        return round(max(0.0, min(1.0, v)), 4)
    try:
        return [cl(float(x) / W), cl(float(y) / H), cl(float(w) / W), cl(float(h) / H)]
    except Exception:
        return None

def parse_box(v):
    """Accept [x,y,w,h] list or {x,y,w,h} dict -> [x,y,w,h] floats, or None."""
    if isinstance(v, dict):
        try:
            return [float(v["x"]), float(v["y"]), float(v["w"]), float(v["h"])]
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) >= 4:
        try:
            return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
        except Exception:
            return None
    return None

def iou(a, b):
    """IoU of two [x,y,w,h] boxes. 0 if either missing/degenerate."""
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax, ay, aw, ah = a[:4]; bx, by, bw, bh = b[:4]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    iw = max(0.0, min(ax2, bx2) - max(ax, bx))
    ih = max(0.0, min(ay2, by2) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
