from rlm.rlm_repl import RLM_REPL
import random

def generate_massive_context(num_lines: int = 100_000, answer: str = "1298418") -> str:
    print("Generating massive context with 1M lines...")
    
    # Set of random words to use
    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
    
    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))
    
    # Insert the magic number at a random position (somewhere in the middle)
    magic_position = random.randint(40000, 60000)
    lines[magic_position] = f"The magic number is {answer}"
    
    print(f"Magic number inserted at position {magic_position}")
    
    return "\n".join(lines)

def main():
    print("Example of using RLM (REPL) with GPT-5-nano on a needle-in-haystack problem.")
    answer = str(random.randint(100000, 999999))
    context = generate_massive_context(num_lines=100_000, answer=answer)

    rlm = RLM_REPL(
        model="gpt-5-mini",
        recursive_model="gpt-5-mini",
        enable_logging=True,
        max_iterations=3
    )
    query = "I'm looking for a magic number. What is it?"
    result = rlm.completion(context=context, query=query)
    print(f"Result: {result}. Expected: {answer}")

# def main():
#     print("Testing RLM with OpenAI (GPT-4)")
    
#     # Simple context
#     context = """
#     The Python programming language was created by Guido van Rossum.
#     It was first released in 1991.
#     Python is known for its simple and readable syntax.
#     """
    
#     # Initialize RLM with OpenAI models
#     rlm = RLM_REPL(
#         model="gpt-5-mini",                    # Root LLM (planning)
#         recursive_model="gpt-5-mini",  # Sub-LLMs (cheaper for sub-tasks)
#         enable_logging=True,
#         max_iterations=2
#     )
    
#     # Ask a question
#     query = "Who created Python and when was it released?"
    
#     print(f"\nQuery: {query}\n")
#     result = rlm.completion(context=context, query=query)
    
#     print(f"\nAnswer: {result}\n")

if __name__ == "__main__":
    main()