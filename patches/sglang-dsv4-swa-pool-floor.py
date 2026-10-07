"""An SWA pool too small for one prefill chunk should say so, not spin.

Move the guard I first wrote into the generic hybrid sizer over to the one
DeepSeek V4 actually uses. That sizer has its own path, _compute_dsv4_sizes,
and it logs

    DSV4 SWA sizing: mode=ratio (explicit), swa_tokens=512
    DSV4 pool sizes: full=131072, swa=512, c4=32768, ...

for --max-total-tokens 131072 against the launcher's ratio of 0.0042. The
chunk is 2048. What follows is not an error: the scheduler tries to admit the
waiting request, cannot reserve its SWA pages, re-queues it and tries again,
forever -- one core near 200%, the GPU flat at 0%, the kt workers parked in
futex_wait, and not one "Prefill batch" line in the log. It reads as a hung
model. An eight-token prompt wedged as thoroughly as an eight-thousand-token
one, which is the tell: the request never got as far as its own length
mattering.

Holding one chunk plus a page is necessary, not provably sufficient -- 512
tokens is known to wedge and 4352 is known to serve, and nothing here pins the
boundary between them. It is enough to turn a silent spin into a message that
names the two knobs that set the pool.

The generic guard above it stays keyed on sliding_window_size, which V4 leaves
as None; that is why nothing caught this.
"""

import io
import sys

import os

# the tree as argv[1] (source/sglang-dsv41 by default)
TREE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/KTransformers/source/sglang-dsv41")
P = os.path.join(TREE, "python/sglang/srt/model_executor/pool_configurator.py")

# Undo the first attempt: right function-shaped, wrong function.
GENERIC = '''
        # Sparse-attention models (DeepSeek V4) report no sliding window, so the
        # check above never fires for them and the floor is the prefill chunk
        # instead. A pool below it does not fail: the scheduler re-queues the
        # request it cannot seat and spins on one core with the GPU idle and no
        # prefill ever logged, which reads as a hung model rather than a
        # misconfigured pool.
        chunk = get_schedule().chunked_prefill_size
        if chunk and chunk > 0 and swa_tokens < chunk + self._page_size:
            raise ValueError(
                f"SWA pool ({swa_tokens} tokens) is smaller than one prefill "
                f"chunk ({chunk}) plus a page ({self._page_size}), so no "
                f"request can ever be admitted. The pool is "
                f"max_total_tokens ({full_tokens}) * swa_full_tokens_ratio "
                f"({self._swa_full_tokens_ratio}); raise either, or lower "
                f"--chunked-prefill-size."
            )
'''

OLD = '''        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
'''

NEW = '''        # The SWA pool has to hold a whole prefill chunk, or the scheduler can
        # never seat a request: it re-queues the one it cannot admit and spins
        # on a single core with the GPU idle and no prefill ever logged, which
        # looks like a hung model rather than a pool sized too small. V4 leaves
        # sliding_window_size as None, so the generic hybrid guard that would
        # have caught this never fires.
        chunk = get_schedule().chunked_prefill_size
        if chunk and chunk > 0 and swa_tokens < chunk + page_size:
            raise ValueError(
                f"DSV4 SWA pool ({swa_tokens} tokens) is smaller than one "
                f"prefill chunk ({chunk}) plus a page ({page_size}), so no "
                f"request can be admitted. The pool is full_tokens "
                f"({full_token}) * swa_full_tokens_ratio ({self.swa_ratio}); "
                f"raise --swa-full-tokens-ratio or --max-total-tokens, or "
                f"lower --chunked-prefill-size."
            )

        return _DSV4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
'''

src = io.open(P, encoding="utf-8").read()

if GENERIC in src:
    src = src.replace(GENERIC, "")
    print("reverted the generic-sizer guard")
elif "smaller than one prefill " in src:
    sys.exit("generic guard present in an unexpected shape; not touching it")

if "never seat a request" in src:
    print("already applied")
    sys.exit(0)
if src.count(OLD) != 1:
    sys.exit("anchor found %d times" % src.count(OLD))

io.open(P, "w", encoding="utf-8", newline="\n").write(src.replace(OLD, NEW))
print("patched", P)
