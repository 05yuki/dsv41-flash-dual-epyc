"""Post-load TP barrier timeout from the environment
(SGLANG_UNBALANCED_LOAD_TIMEOUT_S, default 480 as upstream).

V4.1 pins 336 GB of expert arenas at load time (KT_GPU_STREAM_EARLY_INIT).
Each rank's registration takes 70-1550 s depending on how fragmented memory
is, and rank 1 can only map rank 0's arenas once they exist. When the two
finish more than 480 s apart, monitored_barrier kills the launch: 09-22 15:03
and 09-23 21:55, each after a 20-minute load. The wait is not a hang, so the
V4.1 launcher gives it 1800 s.

Apply once to source/sglang-dsv41.
"""
from pathlib import Path

F = (Path.home() / "KTransformers/source/sglang-dsv41/python/sglang/srt/"
     "model_executor/model_runner_components/load_model_utils.py")
s = F.read_text()
old = "UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing\n"
if s.count(old) != 1:
    raise SystemExit("anchor found %d times" % s.count(old))
s = s.replace(old, 'UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(\n'
              '    os.environ.get("SGLANG_UNBALANCED_LOAD_TIMEOUT_S", "480")\n'
              ')  # leave more time for post data processing\n')
F.write_text(s)
print("patched", F.name)
