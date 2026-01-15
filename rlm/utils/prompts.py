"""
Example prompt templates for the RLM REPL Client.

Supports multi-depth prompts that inform the LLM about its position in the
recursive hierarchy and whether its sub-LLMs have REPL capabilities.
"""

from typing import Dict

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

# Base system prompt for the REPL environment WITH sub-LLM capabilities
REPL_SYSTEM_PROMPT_WITH_SUB_LLMS = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query(prompt: str) -> str` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. A `llm_batch(prompts: List[str], max_concurrent: int = 10) -> List[str]` function that allows you to query the LLM with MULTIPLE prompts in PARALLEL. This is much faster than calling llm_query in a loop. Use this whenever you need to process multiple chunks or queries.
4. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

{depth_info}

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and querying an LLM over it in PARALLEL using llm_batch:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])

# Build prompts for all sections
prompts = []
headers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    prompts.append(f"Summarize this {{header}} section: {{info}}")
    headers.append(header)

# Query all sections in PARALLEL (much faster than sequential llm_query!)
summaries = llm_batch(prompts)

# Combine results
buffers = [f"{{h}}: {{s}}" for h, s in zip(headers, summaries)]
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

PERFORMANCE TIP: Always prefer `llm_batch()` over looping with `llm_query()` when you have multiple independent queries. For example:
- BAD (sequential, slow): `results = [llm_query(p) for p in prompts]`
- GOOD (parallel, fast): `results = llm_batch(prompts)`

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""

# Base system prompt for the REPL environment WITHOUT sub-LLM capabilities (max_depth=1)
REPL_SYSTEM_PROMPT_NO_SUB_LLMS = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
3. Full Python capabilities for data processing, analysis, and manipulation.

**NOTE**: You do NOT have access to `llm_query()` or `llm_batch()` functions. You must analyze the context and answer the query using only Python code execution.

Make sure to explicitly look through the entire context in REPL before answering your query. Use Python's string processing, regex, and other built-in capabilities to analyze the context.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example:
```repl
# Look at the context
print(context[:1000])  # First 1000 characters
```

```repl
# Process and analyze
import re
sections = context.split("\\n\\n")
for i, section in enumerate(sections[:5]):
    print(f"Section {{i}}: {{section[:200]}}")
```

```repl
# Build your answer
key_findings = []
# ... analyze and extract information ...
final_answer = "Based on my analysis: " + ", ".join(key_findings)
print(final_answer)
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response. Remember to explicitly answer the original query in your final answer.
"""

# Depth-specific information to insert into the prompt
DEPTH_INFO_ROOT_WITH_RECURSIVE_SUBS = """**RECURSION INFO**: You are the ROOT LLM (depth 0). Your sub-LLMs (called via llm_query/llm_batch) also have their own REPL environments and can spawn further sub-LLMs! This means you can delegate complex sub-tasks that require code execution and analysis to your sub-LLMs. Current max_depth is {max_depth}."""

DEPTH_INFO_MID_WITH_RECURSIVE_SUBS = """**RECURSION INFO**: You are a SUB-LLM at depth {depth} of {max_depth}. Your sub-LLMs (called via llm_query/llm_batch) also have their own REPL environments and can spawn further sub-LLMs. You can delegate complex sub-tasks to them."""

DEPTH_INFO_WITH_TERMINAL_SUBS = """**RECURSION INFO**: You are at depth {depth} of {max_depth}. Your sub-LLMs (called via llm_query/llm_batch) are TERMINAL - they will directly answer without code execution. Design your prompts to them as complete, self-contained questions."""

def build_system_prompt(depth: int = 0, max_depth: int = 1) -> list[Dict[str, str]]:
    """
    Build the system prompt with depth-aware information.
    
    Args:
        depth: Current depth level (0 = root)
        max_depth: Maximum recursion depth
        
    Returns:
        List of message dicts for the system prompt
    """
    # Check if this node can spawn sub-LLMs
    can_spawn_sub_llms = depth < max_depth - 1
    
    if not can_spawn_sub_llms:
        # No sub-LLMs available - use the simplified prompt WITHOUT llm_query/llm_batch
        prompt_content = REPL_SYSTEM_PROMPT_NO_SUB_LLMS
    else:
        # Sub-LLMs available - use the full prompt WITH llm_query/llm_batch
        if depth == 0:
            # Root with recursive sub-LLMs
            depth_info = DEPTH_INFO_ROOT_WITH_RECURSIVE_SUBS.format(max_depth=max_depth)
        elif depth < max_depth - 2:
            # Mid-level with recursive sub-LLMs (their children also have REPL)
            depth_info = DEPTH_INFO_MID_WITH_RECURSIVE_SUBS.format(depth=depth, max_depth=max_depth)
        else:
            # Can spawn, but children are terminal
            depth_info = DEPTH_INFO_WITH_TERMINAL_SUBS.format(depth=depth, max_depth=max_depth)
        
        prompt_content = REPL_SYSTEM_PROMPT_WITH_SUB_LLMS.format(depth_info=depth_info)
    
    return [
        {
            "role": "system",
            "content": prompt_content
        },
    ]


# Prompt at every step to query root LM to make a decision
USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original query: \"{query}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:""" 
def next_action_prompt(query: str, iteration: int = 0, final_answer: bool = False) -> Dict[str, str]:
    if final_answer:
        return {"role": "user", "content": "Based on all the information you have, provide a final answer to the user's query."}
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your context yet. Your next action should be to look through, don't just provide a final answer yet.\n\n"
        return {"role": "user", "content": safeguard + USER_PROMPT.format(query=query)}
    else:
        return {"role": "user", "content": "The history before is your previous interactions with the REPL environment. " + USER_PROMPT.format(query=query)}
