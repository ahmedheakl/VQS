"""Shared vLLM plumbing: Qwen3-VL prompt wrappers, image loading and chunked generation.

vLLM preprocesses every request's image before decoding starts, so one generate() call over a few
hundred thousand claim checks would hold that many decoded images at once. generate() feeds the
requests in chunks, builds each chunk on a thread pool, and rgb() decodes each file once: the
checkers ask many claims about one image back to back.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

IMG = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
TXT = "<|im_start|>user\n"
END = "<|im_end|>\n<|im_start|>assistant\n"


@lru_cache(maxsize=512)
def rgb(path):
    """The decoded RGB image at `path`, shared by every request about it."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def generate(llm, items, make_request, sampling, chunk=2048, workers=16):
    """llm.generate([make_request(x) for x in items]) in chunks; outputs keep the items' order."""
    outs = []
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(items), chunk):
            part = list(pool.map(make_request, items[i:i + chunk]))
            got = llm.generate(part, sampling)
            assert len(got) == len(part), f"{len(got)} outputs for {len(part)} requests"
            outs += got
    return outs
