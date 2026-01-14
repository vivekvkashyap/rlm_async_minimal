"""
Simple test to verify batch processing functionality.
"""

import time
from rlm.repl import Sub_RLM, REPLEnv

def test_sub_rlm_batch():
    """Test that Sub_RLM batch_completion works."""
    print("Testing Sub_RLM batch_completion...")
    
    try:
        # Initialize Sub_RLM
        sub_rlm = Sub_RLM(model="gpt-5-nano")
        
        # Create test prompts
        prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "What color is the sky?"
        ]
        
        print(f"Sending {len(prompts)} prompts in parallel...")
        start_time = time.time()
        
        # Test batch completion
        results = sub_rlm.batch_completion(prompts, max_concurrent=3)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
        # Verify results
        assert len(results) == len(prompts), "Result count mismatch"
        
        print("\nResults:")
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"{i+1}. Q: {prompt}")
            print(f"   A: {result[:100]}...")
            print()
        
        print("✓ Sub_RLM batch_completion test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Sub_RLM batch_completion test failed: {e}")
        return False

def test_repl_llm_batch():
    """Test that REPLEnv exposes llm_batch function."""
    print("\nTesting REPLEnv llm_batch function...")
    
    try:
        # Initialize REPL environment
        repl_env = REPLEnv(
            recursive_model="gpt-5-nano",
            context_str="Test context"
        )
        
        # Test that llm_batch is available in globals
        assert 'llm_batch' in repl_env.globals, "llm_batch not found in REPL globals"
        
        # Test executing code that uses llm_batch
        code = """
prompts = ["What is 1+1?", "What is 2+2?"]
results = llm_batch(prompts, max_concurrent=2)
print(f"Got {len(results)} results")
batch_results = results
"""
        
        print("Executing REPL code with llm_batch...")
        start_time = time.time()
        
        result = repl_env.code_execution(code)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
        # Check that batch_results variable was created
        assert 'batch_results' in repl_env.locals, "batch_results not found in REPL locals"
        assert isinstance(repl_env.locals['batch_results'], list), "batch_results is not a list"
        assert len(repl_env.locals['batch_results']) == 2, "batch_results length mismatch"
        
        print("\nREPL Output:")
        print(result.stdout)
        
        print("\n✓ REPLEnv llm_batch test passed!")
        return True
        
    except Exception as e:
        print(f"✗ REPLEnv llm_batch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sequential_vs_parallel():
    """Compare sequential vs parallel execution times."""
    print("\nTesting sequential vs parallel performance...")
    
    try:
        sub_rlm = Sub_RLM(model="gpt-5-nano")
        
        prompts = [
            "Count to 3",
            "Name 3 colors",
            "List 3 animals"
        ]
        
        # Test sequential
        print(f"Running {len(prompts)} prompts sequentially...")
        start_time = time.time()
        sequential_results = [sub_rlm.completion(p) for p in prompts]
        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f} seconds")
        
        # Test parallel
        print(f"Running {len(prompts)} prompts in parallel...")
        start_time = time.time()
        parallel_results = sub_rlm.batch_completion(prompts, max_concurrent=3)
        parallel_time = time.time() - start_time
        print(f"Parallel time: {parallel_time:.2f} seconds")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✓ Parallel processing is significantly faster!")
        else:
            print("⚠ Parallel processing speedup is less than expected (might be due to API rate limiting or small batch size)")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("RLM Batch Processing Tests")
    print("=" * 80)
    print()
    
    # Run tests
    test1 = test_sub_rlm_batch()
    test2 = test_repl_llm_batch()
    test3 = test_sequential_vs_parallel()
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Sub_RLM batch_completion: {'PASS' if test1 else 'FAIL'}")
    print(f"REPLEnv llm_batch: {'PASS' if test2 else 'FAIL'}")
    print(f"Performance comparison: {'PASS' if test3 else 'FAIL'}")
    print()
    
    if all([test1, test2, test3]):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")

if __name__ == "__main__":
    main()

