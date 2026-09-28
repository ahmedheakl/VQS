"""Shared vLLM plumbing: Qwen3-VL prompt wrappers and chunked generation.

vLLM preprocesses every request's image before decoding starts, so one generate() call over a few
hundred thousand claim checks would hold that many decoded images at once. generate() feeds the
requests in chunks and opens each image only when its chunk is built.
"""
IMG = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
TXT = "<|im_start|>user\n"
END = "<|im_end|>\n<|im_start|>assistant\n"


def generate(llm, items, make_request, sampling, chunk=2048):
    """llm.generate([make_request(x) for x in items]) in chunks; outputs keep the items' order."""
    outs = []
    for i in range(0, len(items), chunk):
        part = [make_request(x) for x in items[i:i + chunk]]
        got = llm.generate(part, sampling)
        assert len(got) == len(part), f"{len(got)} outputs for {len(part)} requests"
        outs += got
    return outs
