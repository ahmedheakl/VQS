"""Shared helpers for the sg_gen project: GPU pin, IO, JSON parsing, vLLM engine.

Adapted from ../vqa-eval/src/common.py so this project is self-contained. The vLLM
helpers (load_engine / load_processor / mm_prompt / txt_prompt) are the proven Phase-2
versions, including the max_pixels vision-token cap that keeps high-res images from
blowing past max_model_len and crashing a batch.
"""
import os

# Default GPU work to GPU 7.  Launchers may explicitly select either of the two GPUs
# allocated to this project (6 or 7) before importing this module.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
# honour the caller's HF_HOME; do not pin it to a local path

import re
import json
import glob
import hashlib

def load_config(path="config.yaml"):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)

# ---------- constants ----------
MODELS = ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"]
DOMAINS = ["natural", "charts"]
MODES = ["native", "unified"]

# Repo-relative data paths (all scripts run from the sg_gen/ root).
GQA_SG = "gqa/val_sceneGraphs.json"
GQA_IMAGES = "gqa/images/val"
CHARTQA_ANN = "chartqa_complete/data/test/annotations"
CHARTQA_PNG = "chartqa_complete/data/test/png"

EVAL_SETS = "data/eval_sets"
PREDICTIONS = "data/predictions"
RESULTS = "data/results"

# ---------- io ----------
def read_json(path):
    with open(path) as f:
        return json.load(f)

def write_json(path, obj, indent=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def model_tag(model):
    """'Qwen/Qwen3-VL-2B-Instruct' -> 'qwen3vl-2b' for use in output paths."""
    name = model.split("/")[-1].lower()
    name = name.replace("qwen3-vl-", "qwen3vl-").replace("-instruct", "")
    return name

def stable_hash(s):
    return hashlib.md5(str(s).encode()).hexdigest()

# ---------- json parsing ----------
def extract_json(txt):
    """Return the first balanced {...} JSON object in txt, or None."""
    i = txt.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(txt)):
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[i:j + 1])
                except Exception:
                    return None
    return None

def robust_json(text):
    """Parse a JSON object from model text, repairing truncated/malformed output.

    Dense images (charts, busy scenes) can make a small model hit the token cap
    mid-object; json_repair recovers the complete-prefix content instead of dropping
    the whole prediction. Returns a dict or None.
    """
    g = extract_json(text)
    if isinstance(g, dict) and g:
        return g
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i = t.find("{")
    if i < 0:
        return None
    try:
        import json_repair
        g = json_repair.loads(t[i:])
    except Exception:
        g = None
    return g if isinstance(g, dict) and g else None

# ---------- vLLM helpers ----------
def load_engine(model=MODELS[0], max_model_len=16384):
    from vllm import LLM
    # Keep the archived evaluation default for exact reproducibility.  Throughput
    # runs can explicitly raise this on the workstation's dedicated 96-GiB GPUs.
    memory_util = float(os.environ.get("VQS_GPU_MEMORY_UTILIZATION", "0.90"))
    return LLM(model=model, limit_mm_per_prompt={"image": 1}, dtype="bfloat16",
               gpu_memory_utilization=memory_util, max_model_len=max_model_len,
               trust_remote_code=True)

def load_processor(model=MODELS[0]):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model, trust_remote_code=True)

def model_family(model):
    """Infer the processor family from model metadata, not only its path.

    Exported checkpoints in this project are named ``qa_*_merged``.  Looking only at that
    directory name misclassified every merged Qwen checkpoint as ``generic``.  The generic
    path merges the system prompt into the user turn and lets the processor use its uncapped
    default image resolution, while the Qwen path keeps the system turn and applies the
    experiment's explicit ``max_pixels`` cap.  That silently made base-vs-tuned evaluation
    protocols differ and also forced slow serial conversation rendering inside vLLM.

    Fast-path known Hub IDs by name; for local/exported models, inspect ``config.json``.
    """
    m = str(model).lower()
    if "qwen" in m:
        return "qwen"
    if "gemma" in m:
        return "gemma"

    cfg_path = os.path.join(os.fspath(model), "config.json")
    if os.path.isfile(cfg_path):
        try:
            cfg = read_json(cfg_path)
            model_type = str(cfg.get("model_type", "")).lower()
            arch = " ".join(map(str, cfg.get("architectures") or [])).lower()
            meta = f"{model_type} {arch}"
            if "qwen" in meta:
                return "qwen"
            if "gemma" in meta:
                return "gemma"
        except (OSError, ValueError, TypeError):
            pass
    return "generic"

def _img_data_uri(path):
    import base64
    ext = "png" if path.lower().endswith("png") else "jpeg"
    with open(path, "rb") as f:
        return f"data:image/{ext};base64," + base64.b64encode(f.read()).decode()

def chat_msg(image_path, system, user):
    """A single-turn multimodal conversation for llm.chat(). Merges system into the user
    turn (Gemma has no system role) and passes the image as a base64 data URI — model-agnostic,
    so vLLM applies the model's own chat template + image processor."""
    text = (system + "\n\n" + user) if system else user
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _img_data_uri(image_path)}},
        {"type": "text", "text": text}]}]

def mm_prompt(processor, image, system, user, max_pixels=2048 * 28 * 28):
    # max_pixels caps the image's vision-token count (~pixels/(28*28)); without it a
    # high-res image can exceed max_model_len and crash the whole batch.
    from qwen_vl_utils import process_vision_info
    img_item = {"type": "image", "image": image}
    if max_pixels:
        img_item["max_pixels"] = max_pixels
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [img_item, {"type": "text", "text": user}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    return {"prompt": prompt, "multi_modal_data": {"image": imgs}}

def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]
