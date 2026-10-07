"""KT_STREAM_MEMOPS=1: hand MoE work between the GPU stream and kt-kernel with
stream memory operations instead of cudaLaunchHostFunc.

Decode profile, Vision-Exp, 09-23: every MoE layer stops the stream ~44 us at
the submit host function and again at the sync one, because a host function
blocks the stream until the driver's callback thread has run it, and that
thread competes with 64 busy kt workers. Measured in isolation (tools/
stream-memop-probe.cu) a host-function pair costs 23.8 us inside a CUDA graph,
a write-value/wait-value pair 8.2 us, with no stale reads of CPU-written data
over 4,040 round trips.

  submit: the host call arms an entry {func, args} and records
          cuStreamWriteValue32(req[i], 1); a dedicated poller thread sees the
          flag and calls func(args) -- the very call the host function made.
  sync:   the host call arms {sync, allow_n_pending} and records
          write(req[j], 1), wait(done[j] == 1), write(done[j], 0); the poller
          runs TaskQueue::sync(allow) and then raises done[j].

Entries armed while the stream is capturing a CUDA graph are permanent (the
graph replays the same flags); the rest are one-shot on a ring. Flags live in
mapped pinned memory allocated when CPUInfer is built, before any capture.
Nothing here touches the arithmetic.

KT_TASKQUEUE_SPIN_US=<us> (also here): the TaskQueue worker and sync() spin
that long before falling back to the condition variable, removing a futex
wake from each hand-over. Default 0 = unchanged.

Apply once per kt-kernel tree, then rebuild the venvs built from it:
  kt-stream-memops.py                                  source/ktransformers-gemma4/kt-kernel
  kt-stream-memops.py ~/KTransformers/source/ktransformers/kt-kernel
The second tree has no KT_TASKQUEUE_TIMING patch, so its TaskQueue anchors
differ; the spin uses its own clock in both.
"""
import sys
from pathlib import Path

ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        Path.home() / "KTransformers/source/ktransformers-gemma4/kt-kernel") / "cpu_backend"
if "memops_init" in (ROOT / "cpuinfer.h").read_text():
    print("already patched cpuinfer.h")
    raise SystemExit(0)
TIMED = "int timing_every = 0;" in (ROOT / "task_queue.h").read_text()


def sub(name, old, new):
    p = ROOT / name
    s = p.read_text()
    if s.count(old) != 1:
        raise SystemExit("%s: anchor found %d times:\n%s" % (name, s.count(old), old[:160]))
    p.write_text(s.replace(old, new))
    print("patched", name)


# ---------------------------------------------------------------- task queue
if TIMED:
    sub("task_queue.h", '''  int timing_every = 0;
''', '''  int timing_every = 0;
  // KT_TASKQUEUE_SPIN_US: spin this long before a condition-variable wait.
  int spin_us = 0;
  static uint64_t spin_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }
''')
    sub("task_queue.cpp", '''  if (const char* e = getenv("KT_TASKQUEUE_TIMING")) timing_every = atoi(e);
''', '''  if (const char* e = getenv("KT_TASKQUEUE_TIMING")) timing_every = atoi(e);
  if (const char* e = getenv("KT_TASKQUEUE_SPIN_US")) spin_us = atoi(e);
''')
else:
    sub("task_queue.h", '''  std::exception_ptr first_exception;
''', '''  std::exception_ptr first_exception;
  // KT_TASKQUEUE_SPIN_US: spin this long before a condition-variable wait.
  int spin_us = 0;
  static uint64_t spin_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }
''')
    sub("task_queue.h", "#include <vector>\n", "#include <vector>\n#include <chrono>\n#include <cstdint>\n")
    sub("task_queue.cpp", '''TaskQueue::TaskQueue() : done(false), pending(0) {
''', '''TaskQueue::TaskQueue() : done(false), pending(0) {
  if (const char* e = std::getenv("KT_TASKQUEUE_SPIN_US")) spin_us = std::atoi(e);
''')
    sub("task_queue.cpp", "#include <chrono>\n", "#include <chrono>\n#include <cstdlib>\n")

sub("task_queue.cpp", '''  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire);
    });
''', '''  if (spin_us > 0) {
    const uint64_t until = spin_now_ns() + uint64_t(spin_us) * 1000;
    while (pending.load(std::memory_order_acquire) > allow_n_pending && spin_now_ns() < until) {
    }
  }
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire);
    });
''')

sub("task_queue.cpp", '''    } else {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
''', '''    } else {
      if (spin_us > 0) {
        const uint64_t until = spin_now_ns() + uint64_t(spin_us) * 1000;
        while (curr->next.load(std::memory_order_acquire) == nullptr &&
               !done.load(std::memory_order_acquire) && spin_now_ns() < until) {
        }
        if (curr->next.load(std::memory_order_acquire) != nullptr) continue;
      }
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
''')

# ------------------------------------------------------------------ cpuinfer
sub("cpuinfer.h", '''#include <atomic>
#include <condition_variable>
''', '''#include <dlfcn.h>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
''')

for ctor in ('''    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
''', '''    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
''', '''    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
'''):
    sub("cpuinfer.h", ctor, ctor + '''    memops_init();
''')

sub("cpuinfer.h", '''  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\\n", (intptr_t)this);
''', '''  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\\n", (intptr_t)this);
    if (mpoller_.joinable()) {
      mstop_.store(true);
      mpoller_.join();
    }
''')

sub("cpuinfer.h", '''    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
''', '''    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    if (mflags_ != nullptr) {
      const int idx = memop_arm((cudaStream_t)user_cuda_stream, func, args, false, 0);
      write32_((void*)user_cuda_stream, req_dev(idx), 1, 0);
      return;
    }
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
''')

sub("cpuinfer.h", '''    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
''', '''    if (mflags_ != nullptr) {
      const int idx = memop_arm((cudaStream_t)user_cuda_stream, nullptr, nullptr, true, allow_n_pending);
      write32_((void*)user_cuda_stream, req_dev(idx), 1, 0);
      wait32_((void*)user_cuda_stream, done_dev(idx), 1, 0x1 /* CU_STREAM_WAIT_VALUE_EQ */);
      write32_((void*)user_cuda_stream, done_dev(idx), 0, 0);
      return;
    }
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
''')

sub("cpuinfer.h", ''' public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;
};
''', ''' public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;

 private:
  // ---- KT_STREAM_MEMOPS (native-ubuntu/patches/kt-stream-memops.py) ----
  struct MemopEntry {
    void (*func)(void*);
    void* args;
    size_t allow;
    int idx;
    bool sync;
    bool permanent;
  };
  typedef int (*StreamValueFn)(void*, unsigned long long, unsigned int, unsigned int);
  static constexpr int kRing = 4096, kPerm = 4096, kStride = 16;  // 16 x u32 = one cache line
  StreamValueFn write32_ = nullptr, wait32_ = nullptr;
  volatile uint32_t* mflags_ = nullptr;
  uint32_t* mflags_dev_ = nullptr;
  std::atomic<int> ring_next_{0}, perm_next_{0};
  std::mutex mmtx_;
  std::vector<MemopEntry> mentries_;
  std::atomic<uint64_t> mversion_{0};
  std::thread mpoller_;
  std::atomic<bool> mstop_{false};

  volatile uint32_t* req_host(int i) { return mflags_ + (size_t)i * 2 * kStride; }
  volatile uint32_t* done_host(int i) { return mflags_ + (size_t)i * 2 * kStride + kStride; }
  unsigned long long req_dev(int i) { return (unsigned long long)(mflags_dev_ + (size_t)i * 2 * kStride); }
  unsigned long long done_dev(int i) {
    return (unsigned long long)(mflags_dev_ + (size_t)i * 2 * kStride + kStride);
  }

  void memops_init() {
#if defined(KTRANSFORMERS_USE_CUDA)
    const char* on = getenv("KT_STREAM_MEMOPS");
    if (on == nullptr || on[0] != '1') return;
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (h != nullptr) {
      write32_ = (StreamValueFn)dlsym(h, "cuStreamWriteValue32_v2");
      wait32_ = (StreamValueFn)dlsym(h, "cuStreamWaitValue32_v2");
    }
    const size_t bytes = (size_t)(kRing + kPerm) * 2 * kStride * sizeof(uint32_t);
    void* host = nullptr;
    void* dev = nullptr;
    if (write32_ == nullptr || wait32_ == nullptr ||
        cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess) {
      printf("[kt-memops] unavailable, keeping host functions\\n");
      return;
    }
    std::memset(host, 0, bytes);
    mflags_dev_ = (uint32_t*)dev;
    mflags_ = (volatile uint32_t*)host;
    mpoller_ = std::thread(&CPUInfer::memop_poller, this);
    printf("[kt-memops] CPUInfer[0x%lx]: stream memops on, %d ring + %d graph slots\\n", (intptr_t)this, kRing,
           kPerm);
#endif
  }

  int memop_arm(cudaStream_t st, void (*func)(void*), void* args, bool sync, size_t allow) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cs);
    const bool perm = cs == cudaStreamCaptureStatusActive;
    int idx;
    if (perm) {
      const int n = perm_next_.fetch_add(1);
      if (n >= kPerm) throw std::runtime_error("KT_STREAM_MEMOPS: out of graph slots");
      idx = kRing + n;
    } else {
      idx = ring_next_.fetch_add(1) % kRing;
    }
    std::lock_guard<std::mutex> g(mmtx_);
    mentries_.push_back(MemopEntry{func, args, allow, idx, sync, perm});
    mversion_.fetch_add(1, std::memory_order_release);
    return idx;
  }

  void memop_poller() {
    std::vector<MemopEntry> snap;
    uint64_t seen = ~0ull;
    while (!mstop_.load(std::memory_order_relaxed)) {
      const uint64_t v = mversion_.load(std::memory_order_acquire);
      if (v != seen) {
        std::lock_guard<std::mutex> g(mmtx_);
        snap = mentries_;
        seen = mversion_.load(std::memory_order_acquire);
      }
      std::vector<int> retire;
      // Arming order is stream order, so within one scan a submit is handled
      // before the sync that follows it.
      for (const MemopEntry& e : snap) {
        if (*req_host(e.idx) != 1) continue;
        *req_host(e.idx) = 0;
        if (!e.sync) {
          e.func(e.args);
        } else {
          task_queue_->sync(e.allow);
          std::atomic_thread_fence(std::memory_order_release);
          *done_host(e.idx) = 1;
        }
        if (!e.permanent) retire.push_back(e.idx);
      }
      __builtin_ia32_pause();
      if (!retire.empty()) {
        std::lock_guard<std::mutex> g(mmtx_);
        for (int idx : retire) {
          for (size_t i = 0; i < mentries_.size(); i++) {
            if (!mentries_[i].permanent && mentries_[i].idx == idx) {
              mentries_.erase(mentries_.begin() + i);
              break;
            }
          }
        }
        mversion_.fetch_add(1, std::memory_order_release);
      }
    }
  }
};
''')
print("done")
