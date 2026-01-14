"""
SIMPLEST TEST: Prove llm_batch() runs in parallel.
Just times sequential vs parallel - clear and fast.
"""

import time
from rlm.repl import Sub_RLM

# Create Sub_RLM instance
print("Initializing Sub_RLM with gpt-3.5-mini...")
sub_rlm = Sub_RLM(model="gpt-3.5-mini")

# 5 simple questions
prompts = [
    "Say 'ONE' in one word.",
    "Say 'TWO' in one word.",
    "Say 'THREE' in one word.",
    "Say 'FOUR' in one word.",
    "Say 'FIVE' in one word.",
]

print(f"\nTesting with {len(prompts)} prompts\n")

# SEQUENTIAL TEST
print("=" * 60)
print("SEQUENTIAL (one at a time):")
print("=" * 60)
start = time.time()
for i, p in enumerate(prompts, 1):
    print(f"  Call {i}...", end=" ", flush=True)
    result = sub_rlm.completion(p)
    print(f"Done: {result.strip()}")
sequential_time = time.time() - start
print(f"\n⏱️  Total time: {sequential_time:.2f} seconds\n")

# PARALLEL TEST
print("=" * 60)
print("PARALLEL (all at once with llm_batch):")
print("=" * 60)
print(f"  Calling llm_batch() with {len(prompts)} prompts...", flush=True)
start = time.time()
results = sub_rlm.batch_completion(prompts, max_concurrent=5)
parallel_time = time.time() - start
print(f"  ✓ All calls completed!")
for i, r in enumerate(results, 1):
    print(f"    Result {i}: {r.strip()}")
print(f"\n⏱️  Total time: {parallel_time:.2f} seconds\n")

# RESULTS
print("=" * 60)
print("RESULTS:")
print("=" * 60)
speedup = sequential_time / parallel_time
print(f"  Sequential: {sequential_time:.2f}s")
print(f"  Parallel:   {parallel_time:.2f}s")
print(f"  Speedup:    {speedup:.2f}x faster")
print(f"  Saved:      {sequential_time - parallel_time:.2f}s")
print("\n" + ("✅ PARALLEL WORKS!" if speedup > 1.5 else "⚠️  Unexpected result"))
print("=" * 60)

