"""
Simple test to demonstrate parallel speedup of llm_batch() vs sequential llm_query().
This will clearly show the time difference between parallel and sequential execution.
"""

import time
from rlm.repl import Sub_RLM

def test_sequential_vs_parallel():
    """
    Test the speedup of parallel batch processing vs sequential calls.
    """
    print("=" * 80)
    print("TESTING: Sequential vs Parallel LLM Calls")
    print("=" * 80)
    
    # Initialize Sub_RLM (handles sub-LLM calls)
    sub_rlm = Sub_RLM(model="gpt-5-mini")  # Using cheaper/faster model
    
    # Create 5 simple prompts (each takes ~1-2 seconds)
    prompts = [
        "What is 2+2? Answer in one word.",
        "What is the capital of France? Answer in one word.",
        "What color is the sky? Answer in one word.",
        "What is 10*5? Answer in one word.",
        "What is the opposite of hot? Answer in one word.",
    ]
    
    num_prompts = len(prompts)
    
    print(f"\nNumber of prompts: {num_prompts}")
    print(f"Model: gpt-5-mini")
    print("\n" + "-" * 80)
    
    # TEST 1: Sequential execution (one at a time)
    print("\n🐢 TEST 1: SEQUENTIAL EXECUTION (one at a time)")
    print("-" * 80)
    
    start_sequential = time.time()
    sequential_results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{num_prompts}] Calling llm_query()...", end=" ", flush=True)
        result = sub_rlm.completion(prompt)
        sequential_results.append(result)
        print(f"✓ Got: {result.strip()}")
    
    time_sequential = time.time() - start_sequential
    
    print(f"\n⏱️  Sequential total time: {time_sequential:.2f} seconds")
    print(f"📊 Average per call: {time_sequential/num_prompts:.2f} seconds")
    
    # TEST 2: Parallel execution (all at once)
    print("\n" + "-" * 80)
    print("\n🚀 TEST 2: PARALLEL EXECUTION (all at once with llm_batch)")
    print("-" * 80)
    print(f"  Calling llm_batch() with {num_prompts} prompts in parallel...")
    
    start_parallel = time.time()
    parallel_results = sub_rlm.batch_completion(prompts, max_concurrent=num_prompts)
    time_parallel = time.time() - start_parallel
    
    print(f"  ✓ All {num_prompts} calls completed!")
    
    for i, result in enumerate(parallel_results, 1):
        print(f"  [{i}] {result.strip()}")
    
    print(f"\n⏱️  Parallel total time: {time_parallel:.2f} seconds")
    print(f"📊 Average per call: {time_parallel/num_prompts:.2f} seconds")
    
    # COMPARISON
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 80)
    
    speedup = time_sequential / time_parallel
    time_saved = time_sequential - time_parallel
    percent_saved = (time_saved / time_sequential) * 100
    
    print(f"\n  Sequential time: {time_sequential:.2f}s")
    print(f"  Parallel time:   {time_parallel:.2f}s")
    print(f"  Time saved:      {time_saved:.2f}s ({percent_saved:.1f}%)")
    print(f"  Speedup:         {speedup:.2f}x faster")
    
    if speedup > 1.5:
        print(f"\n  ✅ SUCCESS! Parallel is {speedup:.2f}x faster than sequential!")
        print(f"  🎯 With {num_prompts} prompts, you saved {time_saved:.2f} seconds!")
    else:
        print(f"\n  ⚠️  Speedup is only {speedup:.2f}x (expected >1.5x for {num_prompts} prompts)")
        print(f"     This might be due to API rate limits or network latency.")
    
    print("\n" + "=" * 80)
    
    # Verify results match
    print("\n🔍 VERIFICATION: Do sequential and parallel give same results?")
    print("-" * 80)
    for i, (seq, par) in enumerate(zip(sequential_results, parallel_results), 1):
        match = "✓" if seq.strip().lower() == par.strip().lower() else "✗"
        print(f"  [{i}] {match} Sequential: {seq.strip():15} | Parallel: {par.strip()}")
    
    print("\n" + "=" * 80)
    print("🎉 TEST COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("  LLM_BATCH PARALLEL SPEEDUP TEST")
    print("  Testing async batch processing with OpenAI API")
    print("=" * 80)
    print("\nThis test will:")
    print("  1. Make 5 LLM calls sequentially (one at a time)")
    print("  2. Make the same 5 calls in parallel (all at once)")
    print("  3. Compare the execution times")
    print("\nExpected result: Parallel should be 3-5x faster!")
    print("\n" + "=" * 80)
    
    input("\nPress Enter to start the test (this will use OpenAI API credits)...")
    
    test_sequential_vs_parallel()

