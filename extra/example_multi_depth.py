"""
Example demonstrating multi-depth recursion in RLM.

This example shows how the RLM can spawn sub-RLMs that also have REPL
environments and can spawn their own sub-RLMs.

Depth Hierarchy (with max_depth=3):
- Depth 0 (Root): Has REPL, can call sub-LLMs
- Depth 1 (Sub): Has REPL, can call sub-LLMs  
- Depth 2 (Sub-sub): Has REPL, calls terminal Sub_RLM (no REPL)

Use Cases:
- Complex multi-step analysis where sub-tasks need their own code execution
- Hierarchical document processing
- Divide-and-conquer algorithms on text
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm.rlm_repl import RLM_REPL


def example_depth_2():
    """
    Example with max_depth=2.
    
    Root (depth=0) can call sub-LLMs that have their own REPL (depth=1).
    Those sub-LLMs call terminal sub-LLMs (depth=2, no REPL).
    """
    print("=" * 60)
    print("Example: max_depth=2")
    print("=" * 60)
    
    # Create a context with nested complexity
    context = """
    MASTER TASK: Analyze the following three research areas and provide a synthesis.
    
    AREA 1 - Machine Learning:
    Deep learning has revolutionized computer vision. CNNs excel at image recognition.
    Transformers have become dominant in NLP. Attention mechanisms enable parallel processing.
    
    AREA 2 - Distributed Systems:
    Microservices architecture enables scalability. Kubernetes orchestrates containers.
    Event-driven systems provide loose coupling. Message queues ensure reliability.
    
    AREA 3 - Security:
    Zero-trust architecture assumes no implicit trust. Multi-factor authentication adds layers.
    Encryption at rest and in transit protects data. Regular audits ensure compliance.
    
    For each area, identify the key theme and main technologies.
    Then synthesize how these three areas could work together in a modern application.
    """
    
    # Create RLM with depth=2
    # Root (depth=0) -> Sub with REPL (depth=1) -> Terminal (no REPL)
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        max_depth=2,  # Enable 2 levels of recursion
        max_iterations=5,
        enable_logging=True,
    )
    
    query = "Analyze each research area using sub-LLMs and synthesize the findings."
    
    result = rlm.completion(context=context, query=query)
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(result)
    
    return result


def example_depth_3():
    """
    Example with max_depth=3 for deeper recursion.
    
    Root (depth=0) -> Sub with REPL (depth=1) -> Sub with REPL (depth=2) -> Terminal (depth=3)
    """
    print("=" * 60)
    print("Example: max_depth=3 (deeper recursion)")
    print("=" * 60)
    
    # Create a more complex context that benefits from deeper recursion
    context = """
    PROJECT ANALYSIS REQUEST
    
    Analyze this software project structure and provide recommendations:
    
    ## Frontend Module
    - React components with TypeScript
    - Redux for state management
    - Material UI for components
    - Jest for unit testing
    
    ## Backend Module
    - FastAPI with Python
    - PostgreSQL database
    - Redis for caching
    - Celery for async tasks
    
    ## Infrastructure Module
    - Docker containers
    - Kubernetes deployment
    - Terraform for IaC
    - GitHub Actions for CI/CD
    
    ## Security Module
    - OAuth 2.0 authentication
    - JWT token management
    - Rate limiting
    - Input validation
    
    For each module:
    1. Identify strengths and weaknesses
    2. Suggest improvements
    3. Rate the overall maturity (1-10)
    
    Then provide an overall project assessment.
    """
    
    # Create RLM with depth=3 for deeper analysis
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        max_depth=3,  # Enable 3 levels of recursion
        max_iterations=5,
        enable_logging=True,
    )
    
    query = "Perform a comprehensive analysis of each module using recursive sub-LLMs."
    
    result = rlm.completion(context=context, query=query)
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(result)
    
    return result


def example_compare_depths():
    """
    Compare behavior with different max_depth values.
    """
    print("=" * 60)
    print("Comparing different max_depth values")
    print("=" * 60)
    
    context = "What is 2 + 2? Explain step by step."
    query = "Answer the question."
    
    for depth in [1, 2]:
        print(f"\n--- max_depth={depth} ---")
        rlm = RLM_REPL(
            model="gpt-4o-mini",
            recursive_model="gpt-4o-mini",
            max_depth=depth,
            max_iterations=3,
            enable_logging=False,  # Quiet mode for comparison
        )
        result = rlm.completion(context=context, query=query)
        print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-depth RLM examples")
    parser.add_argument("--depth", type=int, default=2, choices=[2, 3],
                        help="Which depth example to run (2 or 3)")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison of different depths")
    
    args = parser.parse_args()
    
    if args.compare:
        example_compare_depths()
    elif args.depth == 2:
        example_depth_2()
    elif args.depth == 3:
        example_depth_3()

