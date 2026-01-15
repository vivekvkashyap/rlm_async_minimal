"""
Example demonstrating RLM with execution trace logging for visualization.

This example runs an RLM with trace logging enabled, then you can visualize
the execution tree using: streamlit run visualize_traces.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm.rlm_repl import RLM_REPL
from rlm.logger.trace_logger import init_trace_logger, get_trace_logger


def example_with_trace():
    """
    Example with max_depth=2 and trace logging enabled.
    """
    print("=" * 60)
    print("Example: RLM with Execution Trace Logging")
    print("=" * 60)
    
    # Initialize trace logger BEFORE creating RLM
    trace_logger = init_trace_logger(output_dir="traces", enabled=True)
    print("✅ Trace logger initialized")
    
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
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        max_depth=2,  # Enable 2 levels of recursion
        max_iterations=3,  # Fewer iterations for faster demo
        enable_logging=True,
    )
    
    query = "Analyze each research area using sub-LLMs and synthesize the findings."
    
    print("\n🚀 Starting RLM execution...")
    result = rlm.completion(context=context, query=query)
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(result)
    
    # Save trace to JSON file
    trace_file = trace_logger.save_trace()
    
    print("\n" + "=" * 60)
    print("✅ Execution complete!")
    print("=" * 60)
    print(f"\n📊 Trace file saved to: {trace_file}")
    print("\n🎨 To visualize the execution tree, run:")
    print(f"    streamlit run visualize_traces.py")
    print("\n💡 This will open an interactive dashboard showing:")
    print("   - Execution tree with parent-child relationships")
    print("   - Click on nodes to see prompts, responses, and code")
    print("   - Timeline of iterations and sub-LLM spawns")
    print("   - Performance statistics")
    
    return result


if __name__ == "__main__":
    example_with_trace()

