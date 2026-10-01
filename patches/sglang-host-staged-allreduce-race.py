"""Fix a race in the host-staged all-reduce (SGLANG_HOST_STAGED_ALLREDUCE=1,
messages of 1 MiB and up): the sum was added in place into the input before
this rank's own copy of that input had left for host memory.

  d2h stream:  src -> pinned host, then raise our counter
  h2d stream:  wait for the PEER's counter, peer's piece -> tmp,
               src += tmp                                  <- in place
  return:      current stream waits for h2d only

Nothing ordered the in-place add after our own D2H copies, so when the peer's
counter came first the peer could read a src that already held part of the
sum, and the caller's next kernels could overwrite src while d2h was still
reading it. The result was a wrong, run-to-run different sum. 09-26, Flash-Next
served logprobs: prompts of 224+ tokens (hidden 2560 bf16 = 1.15 MB, over the
1 MiB threshold) had median |d logprob| 0.1 nat between identical requests,
max 2-5 nat; 20-192 tokens were bit-identical.

Fix: the h2d stream waits for the d2h stream before the add (and so does the
caller, through h2d). The copies still overlap; only the add waits.

Apply to every tree that carries host_staged_allreduce.py:
  sglang-host-staged-allreduce-race.py source/sglang-dsv41 source/sglang-upstream \
      source/sglang-qwen38-next-nvidia-v0519 source/sglang-qwen38-flash-next
"""
import sys
from pathlib import Path

OLD = '''                src.view(t.dtype).add_(self.tmp[:n].view(t.dtype))
'''
NEW = '''                # Our own pieces must have left src before the add overwrites
                # it; otherwise the peer can DMA a half-summed src (09-26,
                # patches/sglang-host-staged-allreduce-race.py).
                self.h2d.wait_stream(self.d2h)
                src.view(t.dtype).add_(self.tmp[:n].view(t.dtype))
'''

trees = sys.argv[1:] or [str(Path.home() / "KTransformers/source/sglang-qwen38-next-nvidia-v0519")]
for tree in trees:
    f = Path(tree) / "python/sglang/srt/distributed/device_communicators/host_staged_allreduce.py"
    s = f.read_text()
    if "self.h2d.wait_stream(self.d2h)" in s:
        print("already patched", f)
        continue
    if s.count(OLD) != 1:
        raise SystemExit("anchor found %d times in %s" % (s.count(OLD), f))
    f.write_text(s.replace(OLD, NEW))
    print("patched", f)
