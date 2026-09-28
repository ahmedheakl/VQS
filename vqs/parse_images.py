"""Stage 1 of the loop: model parses each image into a structure. Label-free.

One JSON schema per domain, verbose keys (the sufficiency study showed the 2B loses
11.7 points on GQA when keys are compressed). vLLM structured outputs force valid JSON,
so the >=98% valid-JSON gate is satisfied by construction rather than by prompting.

Usage: CUDA_VISIBLE_DEVICES=7 python scripts/parse_images.py --manifest runs/pool/manifest.jsonl \
         --out runs/parse/parses.jsonl [--limit 32]
"""
import argparse
import json
import os
import pathlib

# honour the caller's HF_HOME; do not pin it to a local path

MODEL = "Qwen/Qwen3-VL-2B-Instruct"

NUM = {"type": ["number", "null"]}
SCHEMAS = {
    "charts": {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "title", "x_axis_title", "y_axis_title", "series"],
        "properties": {
            "kind": {"const": "chart"},
            "title": {"type": "string"},
            "x_axis_title": {"type": "string"},
            "y_axis_title": {"type": "string"},
            "series": {"type": "array", "maxItems": 6, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["series_name", "points"],
                "properties": {
                    "series_name": {"type": "string"},
                    "points": {"type": "array", "maxItems": 24, "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["category", "printed_value", "numeric_value"],
                        "properties": {"category": {"type": "string"},
                                       "printed_value": {"type": "string"},
                                       "numeric_value": NUM}}}}}}}},
    "diagrams": {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "title", "has_arrows", "nodes", "edges"],
        "properties": {
            "kind": {"const": "diagram"},
            "title": {"type": "string"},
            # AI2D diagrams are often labelled parts with NO arrows between them. Without
            # this flag the model invents edges: the fact-checker killed 85% of diagram
            # questions in the smoke run. Edge questions now require has_arrows=true.
            "has_arrows": {"type": "boolean"},
            "nodes": {"type": "array", "maxItems": 20, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["node_id", "node_label"],
                "properties": {"node_id": {"type": "integer"},
                               "node_label": {"type": "string"}}}},
            "edges": {"type": "array", "maxItems": 30, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["from_node_id", "to_node_id", "edge_label"],
                "properties": {"from_node_id": {"type": "integer"},
                               "to_node_id": {"type": "integer"},
                               "edge_label": {"type": "string"}}}}}},
    "natural": {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "objects", "relations"],
        "properties": {
            "kind": {"const": "photo"},
            "objects": {"type": "array", "maxItems": 10, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["object_id", "object_name", "attributes", "box_0_1000"],
                "properties": {
                    "object_id": {"type": "integer"},
                    "object_name": {"type": "string"},
                    "attributes": {"type": "array", "maxItems": 3,
                                   "items": {"type": "string"}},
                    "box_0_1000": {"type": "array", "minItems": 4, "maxItems": 4,
                                   "items": {"type": "integer"}}}}},
            "relations": {"type": "array", "maxItems": 12, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["subject_id", "predicate", "object_id"],
                "properties": {"subject_id": {"type": "integer"},
                               "predicate": {"type": "string"},
                               "object_id": {"type": "integer"}}}}}},
    "infographics": {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "title", "entries"],
        "properties": {
            "kind": {"const": "infographic"},
            "title": {"type": "string"},
            "entries": {"type": "array", "maxItems": 20, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["entry_label", "printed_value", "numeric_value", "unit"],
                "properties": {"entry_label": {"type": "string"},
                               "printed_value": {"type": "string"},
                               "numeric_value": NUM,
                               "unit": {"type": "string"}}}}}},
}

PROMPTS = {
    "charts": ("Read this chart. List every series and every data point you can see. "
               "Copy `printed_value` exactly as printed on the chart (keep %, $, B, commas). "
               "Put the plain number in `numeric_value`. Use the real category and axis names."),
    "diagrams": ("Read this diagram. List every labelled part as a node, copying labels "
                 "verbatim. Set has_arrows to true ONLY if the diagram really draws arrows "
                 "from one labelled part to another, like a cycle or a food web. Many "
                 "diagrams only label parts and have no such arrows -- for those set "
                 "has_arrows to false and leave edges empty. Never invent an arrow."),
    "natural": ("Describe this photo as structure. List the main objects with their visible "
                "attributes and boxes in 0-1000 coordinates as [x0,y0,x1,y1]. Then list how the "
                "objects relate to each other, using simple predicates like \"on\", \"holding\"."),
    "infographics": ("Read this infographic. List every labelled figure or statistic as an entry. "
                     "Copy `printed_value` exactly as printed (keep %, $, commas, K/M/B). "
                     "Put the plain number in `numeric_value` and the unit in `unit`."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domains", default="", help="comma list; default all")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--model", default=MODEL, help="trained parser, or the base")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest)]
    if a.domains:
        want = set(a.domains.split(","))
        rows = [r for r in rows if r["domain"] in want]
        assert rows, f"no images for domains {want}"
    if a.limit:
        rows = rows[: a.limit]
    assert rows, "empty manifest"
    print(f"parsing {len(rows)} images", flush=True)

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model=a.model, max_model_len=16384, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True,
              max_num_seqs=64, mm_processor_kwargs={"max_pixels": 1003520})

    outdir = pathlib.Path(a.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(a.out, "w") as fout:
        for domain in sorted({r["domain"] for r in rows}):
            sub = [r for r in rows if r["domain"] == domain]
            sp = SamplingParams(temperature=0, max_tokens=a.max_tokens,
                                structured_outputs=StructuredOutputsParams(json=SCHEMAS[domain]))
            reqs = [{"prompt": ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                                f"{PROMPTS[domain]}<|im_end|>\n<|im_start|>assistant\n"),
                     "multi_modal_data": {"image": Image.open(r["image"]).convert("RGB")}}
                    for r in sub]
            outs = llm.generate(reqs, sp)
            assert len(outs) == len(sub), f"{len(outs)} outputs for {len(sub)} inputs"
            n_trunc = 0
            for r, o in zip(sub, outs):
                # structured outputs guarantees a grammar-valid PREFIX, not a finished
                # object -- hitting the token cap yields unparseable JSON. Count, do not hide.
                if o.outputs[0].finish_reason == "length":
                    n_trunc += 1
                    continue
                parse = json.loads(o.outputs[0].text)  # must parse; crash if not
                fout.write(json.dumps({**r, "parse": parse,
                                       "parse_tokens": len(o.outputs[0].token_ids)}) + "\n")
                n_ok += 1
            rate = n_trunc / len(sub)
            print(f"{domain:14s} kept {len(sub)-n_trunc}/{len(sub)}  truncated {n_trunc} "
                  f"({rate:.1%})", flush=True)
            assert rate < 0.05, f"{domain}: {rate:.1%} truncated, raise --max-tokens"
    print(f"WROTE {n_ok} parses -> {a.out}")
    print(f"kept {n_ok}/{len(rows)} = {n_ok/len(rows):.1%}")


if __name__ == "__main__":
    main()
