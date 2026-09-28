"""GQA scene graph -> canonical scored form for natural images.

Canonical form (shared by GT and predictions so metrics are identical):
  {
    "objects":            [name, ...],                 # multiset of object names
    "attributes":         [(obj_name, attr), ...],
    "relations_semantic": [(subj, pred, obj), ...],    # HEADLINE relations
    "relations_spatial":  [(subj, pred, obj), ...],    # auto left/right, scored separately
  }

GQA relations reference the *object id* of the target; we resolve ids -> names here.
~94% of GQA relations are the auto-generated 'to the left/right of' pairs, so they are
split into a separate spatial bucket and kept out of the headline relation metric.
"""
import os
from . import common as c

SPATIAL_PREDS = {"to the left of", "to the right of", "left of", "right of"}


def load_gqa(path=None):
    """Load the full GQA val scene-graph dict {image_id: entry}."""
    return c.read_json(path or c.GQA_SG)

def image_path(image_id, root=None):
    return os.path.join(root or c.GQA_IMAGES, f"{image_id}.jpg")


def canon_natural(entry):
    """GQA scene-graph entry -> canonical form (incl. per-object normalized boxes)."""
    from .bbox import norm_box
    objects = entry.get("objects", {}) or {}
    W, H = entry.get("width"), entry.get("height")
    # id -> name for resolving relation targets
    id2name = {oid: o.get("name", "") for oid, o in objects.items()}

    obj_names, attrs, rel_sem, rel_spa, obj_boxes = [], [], [], [], []
    for oid, o in objects.items():
        name = o.get("name", "")
        if not name:
            continue
        obj_names.append(name)
        obj_boxes.append((name, norm_box(o.get("x", 0), o.get("y", 0), o.get("w", 0), o.get("h", 0), W, H)))
        for a in (o.get("attributes") or []):
            attrs.append((name, a))
        for r in (o.get("relations") or []):
            pred = r.get("name", "")
            tgt = id2name.get(r.get("object", ""), "")
            if not (pred and tgt):
                continue
            bucket = rel_spa if pred.lower() in SPATIAL_PREDS else rel_sem
            bucket.append((name, pred, tgt))
    return {
        "objects": obj_names,
        "attributes": attrs,
        "relations_semantic": rel_sem,
        "relations_spatial": rel_spa,
        "obj_boxes": obj_boxes,
    }
