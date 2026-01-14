"""
Example demonstrating the use of llm_batch for parallel sub-LLM calls.

This example shows how the RLM can now use llm_batch() to process
multiple chunks in parallel instead of sequentially.
"""

from rlm.rlm_repl import RLM_REPL
import random

def generate_context_with_multiple_facts(num_facts: int = 10) -> str:
    """Generate a context with multiple facts scattered throughout."""
    print(f"Generating context with {num_facts} facts...")
    
    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
    
    lines = []
    facts = []
    
    for i in range(num_facts):
        # Add some random lines before each fact
        for _ in range(100):
            num_words = random.randint(3, 8)
            line_words = [random.choice(random_words) for _ in range(num_words)]
            lines.append(" ".join(line_words))
        
        # Insert a fact
        fact = f"FACT_{i+1}: The answer to question {i+1} is {random.randint(100, 999)}"
        facts.append(fact)
        lines.append(fact)
    
    print(f"Facts inserted: {facts[:3]}... (showing first 3)")
    
    return "\n".join(lines)

def main():
    print("=" * 80)
    print("Example: Using llm_batch for parallel sub-LLM processing")
    print("=" * 80)
    
    # Generate context with multiple facts
    context = generate_context_with_multiple_facts(num_facts=5)
    
    # Initialize RLM with batch support
    rlm = RLM_REPL(
        model="gpt-5",
        recursive_model="gpt-5-nano",
        enable_logging=True,
        max_iterations=15
    )
    
    # Query that will benefit from parallel processing
    query = "Find all the FACT entries in the context and list them all."
    
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    print("\nThe RLM will now use llm_batch() to process chunks in parallel...")
    print("This is much faster than sequential llm_query() calls!\n")
    
    result = rlm.completion(context=context, query=query)
    
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result)
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

