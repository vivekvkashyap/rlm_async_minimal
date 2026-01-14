# Async Batch Processing Implementation - Changes Summary

## Overview
Added async/parallel batch processing capability to the RLM system, enabling sub-LLM calls to be executed concurrently instead of sequentially using OpenAI's API. This provides significant performance improvements when processing multiple chunks or queries.

## Files Modified

### 1. `rlm/utils/llm.py` ✨ NEW CLASS
**Changes:**
- Added `import asyncio` and `from typing import List`
- Added `AsyncOpenAI` import from openai package
- Created new `AsyncOpenAIClient` class with:
  - `async completion()`: Async single completion
  - `async batch_completion()`: Parallel batch completions with semaphore-based rate limiting
  - `batch_completion_sync()`: Synchronous wrapper for batch completions

**Key Features:**
- Concurrent execution using `asyncio.gather()`
- Configurable `max_concurrent` parameter (default: 10)
- Error handling per request (doesn't fail entire batch)
- Thread-safe event loop management

### 2. `rlm/repl.py` ⚡ ENHANCED
**Changes:**
- Added `from typing import List` import
- Modified `Sub_RLM.__init__()`:
  - Added `AsyncOpenAIClient` import
  - Initialized `self.async_client` alongside existing `self.client`
- Added `batch_completion()` method to `Sub_RLM` class:
  - Takes `List[str]` of prompts
  - Returns `List[str]` of responses
  - Supports `max_concurrent` parameter
- Modified `REPLEnv.__init__()`:
  - Added `llm_batch()` function definition
  - Registered `llm_batch` in `self.globals`

**Key Features:**
- Backward compatible (existing `completion()` unchanged)
- Exposes batch functionality to REPL code
- Maintains order of results

### 3. `rlm/utils/prompts.py` 📝 UPDATED
**Changes:**
- Updated `REPL_SYSTEM_PROMPT`:
  - Added documentation for `llm_batch()` function
  - Updated example code to use `llm_batch()` instead of sequential `llm_query()`
  - Added performance tip comparing sequential vs parallel approaches

**Key Features:**
- Teaches the root LLM about batch processing
- Encourages parallel processing when appropriate
- Provides clear examples

### 4. `requirements.txt` 🔧 FIXED
**Changes:**
- Changed `dotenv` to `python-dotenv` (correct package name)

## Files Added

### 1. `example_batch.py` 📘 NEW
**Purpose:** Demonstration script showing batch processing in action
**Features:**
- Generates context with multiple facts
- Shows how RLM uses `llm_batch()` for parallel processing
- Includes logging to show performance

### 2. `test_batch.py` 🧪 NEW
**Purpose:** Unit tests for batch processing functionality
**Tests:**
- `test_sub_rlm_batch()`: Tests Sub_RLM batch_completion
- `test_repl_llm_batch()`: Tests llm_batch in REPL environment
- `test_sequential_vs_parallel()`: Performance comparison

### 3. `ASYNC_BATCH_GUIDE.md` 📚 NEW
**Purpose:** Comprehensive documentation for async batch processing
**Contents:**
- Overview of changes
- Performance comparisons
- Usage examples
- API reference
- Best practices
- Troubleshooting guide

### 4. `CHANGES_SUMMARY.md` 📋 NEW (this file)
**Purpose:** Summary of all changes made

### 5. `README.md` 📖 UPDATED
**Changes:**
- Added section highlighting new async batch processing feature
- Added link to ASYNC_BATCH_GUIDE.md
- Updated example command to run batch example

## Architecture Changes

### Before (Sequential)
```
Root LLM → generates code with llm_query() calls
  ↓
REPL executes code
  ↓
llm_query("prompt 1") → API Call 1 (2s)
  ↓
llm_query("prompt 2") → API Call 2 (2s)
  ↓
llm_query("prompt 3") → API Call 3 (2s)
  ↓
Total: ~6 seconds
```

### After (Parallel)
```
Root LLM → generates code with llm_batch() call
  ↓
REPL executes code
  ↓
llm_batch([p1, p2, p3]) → ┬→ API Call 1 ┐
                          ├→ API Call 2 ├→ Results (2s)
                          └→ API Call 3 ┘
  ↓
Total: ~2 seconds (3x faster!)
```

## API Changes

### New Functions Available in REPL

#### `llm_batch(prompts: List[str], max_concurrent: int = 10) -> List[str]`
- **Purpose**: Execute multiple LLM queries in parallel
- **Parameters**:
  - `prompts`: List of string prompts
  - `max_concurrent`: Max concurrent requests (default: 10)
- **Returns**: List of string responses (same order as input)
- **Example**:
  ```python
  prompts = ["What is 2+2?", "What is 3+3?"]
  results = llm_batch(prompts)
  ```

### Existing Functions (Unchanged)

#### `llm_query(prompt: str) -> str`
- Still works exactly as before
- Use for single queries or when queries depend on each other

## Performance Impact

### Expected Speedups
- **3 parallel calls**: ~3x faster
- **10 parallel calls**: ~10x faster (with sufficient rate limits)
- **100 parallel calls**: Limited by `max_concurrent` parameter

### Rate Limiting
- Default `max_concurrent=10` is conservative
- Adjust based on your API rate limits
- OpenAI typically allows 10-100+ concurrent requests depending on tier

## Backward Compatibility

✅ **Fully backward compatible**
- All existing code works without changes
- `llm_query()` unchanged
- New `llm_batch()` is opt-in
- No breaking changes to any APIs

## Testing

Run tests to verify functionality:
```bash
# Activate conda environment
conda activate tf-gpu-211

# Run unit tests
python test_batch.py

# Run example
python example_batch.py

# Run original example (still works)
python main.py
```

## Usage Guidelines

### When to Use `llm_batch()`
✅ Processing multiple independent chunks
✅ Parallel summarization of sections
✅ Batch question answering
✅ Any scenario with multiple independent queries

### When to Use `llm_query()`
✅ Single queries
✅ Sequential queries where each depends on previous results
✅ When you need immediate feedback per query

## Technical Implementation Details

### Async Event Loop Management
- Uses `asyncio.run()` for new event loops
- Handles existing event loops with ThreadPoolExecutor
- Thread-safe execution from synchronous REPL code

### Error Handling
- Individual request failures don't crash batch
- Exceptions converted to error strings
- Results always match input length and order

### Rate Limiting
- Semaphore-based concurrency control
- Configurable per-batch via `max_concurrent`
- Prevents overwhelming API servers

## Future Enhancements

Potential improvements for future versions:
- [ ] Exponential backoff retry logic
- [ ] Progress callbacks for long batches
- [ ] Cost tracking for batch operations
- [ ] Streaming responses
- [ ] Full async RLM_REPL class
- [ ] Batch caching/deduplication

## Migration Guide

No migration needed! The changes are additive and fully backward compatible.

To take advantage of batch processing:
1. Update your prompts to use `llm_batch()` when appropriate
2. The root LLM will automatically learn about it from system prompts
3. Existing code continues to work as-is

## Questions?

See [ASYNC_BATCH_GUIDE.md](ASYNC_BATCH_GUIDE.md) for detailed documentation.

---

# Multi-Depth Recursion Implementation - Changes Summary

## Overview
Added multi-depth recursion capability to the RLM system, enabling sub-LLMs to themselves act as RLM_REPL instances with their own REPL environments. This allows for hierarchical problem decomposition where sub-LLMs can spawn further sub-LLMs up to a configurable maximum depth, rather than being limited to terminal API calls.

## Files Modified

### 1. `rlm/rlm_repl.py` 🔄 ENHANCED
**Changes:**
- Added `depth` parameter to `__init__()` (default: 0 for root)
- Added `max_depth` parameter to `__init__()` (default: 1 for backward compatibility)
- Stored `depth` and `max_depth` as instance variables
- Modified `setup_context()` to pass `depth` and `max_depth` to `REPLEnv`
- Adjusted `max_iterations` based on depth to manage cost (deeper levels get fewer iterations)
- Added depth-aware logging with `[Depth {depth}]` prefix
- Updated docstring to explain multi-depth recursion

**Key Features:**
- Root LLM is at depth=0
- Each sub-RLM increments depth by 1
- Iterations are reduced for deeper levels to prevent excessive API costs
- Backward compatible (default `max_depth=1` maintains original behavior)

### 2. `rlm/repl.py` 🔄 ENHANCED
**Changes:**
- Added `depth`, `max_depth`, `max_iterations`, and `enable_logging` parameters to `REPLEnv.__init__()`
- Implemented conditional sub-RLM initialization:
  - If `depth < max_depth - 1`: Creates `RLM_REPL` instance (recursive, has REPL)
  - If `depth >= max_depth - 1`: Creates `Sub_RLM` instance (terminal, no REPL)
- Modified `llm_query()` to handle both `RLM_REPL` and `Sub_RLM` interfaces:
  - `RLM_REPL`: Calls `completion(context=prompt, query=...)`
  - `Sub_RLM`: Calls `completion(prompt)` directly
- Modified `llm_batch()` to handle both interfaces:
  - `RLM_REPL`: Uses `_batch_recursive_completion()` for parallel recursive calls
  - `Sub_RLM`: Uses `batch_completion()` for parallel API calls
- Added `_batch_recursive_completion()` helper method for parallel RLM_REPL calls
- Updated docstrings to explain depth-based behavior

**Key Features:**
- Dynamic sub-RLM type based on recursion depth
- Seamless handling of recursive vs terminal sub-LLMs
- Parallel batch processing works for both recursive and terminal sub-LLMs
- Each recursive sub-RLM gets its own isolated REPL environment

### 3. `rlm/utils/prompts.py` 📝 UPDATED
**Changes:**
- Updated `REPL_SYSTEM_PROMPT` to mention multi-depth recursion capability
- Added note that sub-LLMs can also have REPL environments and call further sub-LLMs
- Added `depth` and `max_depth` parameters to `build_system_prompt()` (for future depth-aware customization)
- Added `depth` parameter to `next_action_prompt()` (for future depth-aware prompting)

**Key Features:**
- System prompt now educates LLM about recursive capabilities
- Foundation for depth-specific prompting strategies

### 4. `rlm/logger/root_logger.py` 📊 UPDATED
**Changes:**
- Added `depth` parameter to `__init__()`
- Stored `depth` as instance variable
- Modified log messages to include `[Depth {self.depth}]` prefix

**Key Features:**
- Depth-aware logging for better traceability
- Easy to track which depth level generated each log message

### 5. `rlm/logger/repl_logger.py` 📊 UPDATED
**Changes:**
- Added `depth` parameter to `__init__()`
- Stored `depth` as instance variable
- Modified log messages to include `[Depth {self.depth}]` prefix

**Key Features:**
- Consistent depth tracking across all loggers
- Helps debug multi-level recursive calls

### 6. `rlm/rlm.py` 🔧 FIXED
**Changes:**
- Fixed Python 3.9 compatibility issue with type annotations
- Changed `list[str] | str | dict[str, str]` to `Union[List[str], str, Dict[str, str]]`
- Added `from typing import Union, List, Dict` imports

**Key Features:**
- Compatible with Python 3.9+
- Maintains type safety

## Files Added

### 1. `extra/example_multi_depth.py` 📘 NEW
**Purpose:** Demonstration script showing multi-depth recursion in action

**Features:**
- Example with `max_depth=2`: Root → Sub-RLM with REPL → Terminal Sub_RLM
- Example with `max_depth=3`: Root → Sub-RLM → Sub-sub-RLM → Terminal Sub_RLM
- Shows hierarchical problem decomposition
- Demonstrates how sub-RLMs can spawn their own sub-RLMs

## Architecture Changes

### Before (Single Depth)
```
Root LLM (depth=0)
  ↓
REPL Environment
  ↓
Sub_RLM (terminal, no REPL)
  ↓
Direct API calls only
```

### After (Multi-Depth)
```
Root LLM (depth=0, max_depth=3)
  ↓
REPL Environment
  ↓
Sub-RLM (depth=1) - Has REPL
  ↓
REPL Environment
  ↓
Sub-sub-RLM (depth=2) - Has REPL
  ↓
REPL Environment
  ↓
Terminal Sub_RLM (depth=3) - No REPL
  ↓
Direct API calls
```

## API Changes

### New Parameters

#### `RLM_REPL.__init__()`
- **`depth: int = 0`**: Current recursion depth (0 for root)
- **`max_depth: int = 1`**: Maximum allowed recursion depth (default: 1 for backward compatibility)

#### `REPLEnv.__init__()`
- **`depth: int = 0`**: Current recursion depth
- **`max_depth: int = 1`**: Maximum allowed recursion depth
- **`max_iterations: int = 20`**: Maximum iterations for sub-RLMs
- **`enable_logging: bool = False`**: Enable depth-aware logging

### Behavior Changes

#### `llm_query()` and `llm_batch()`
- Now handle both recursive (`RLM_REPL`) and terminal (`Sub_RLM`) sub-LLMs
- Automatically detect sub-RLM type and call appropriate interface
- Recursive sub-LLMs receive prompts as `context` parameter
- Terminal sub-LLMs receive prompts directly

## Usage Examples

### Basic Multi-Depth Setup
```python
from rlm.rlm_repl import RLM_REPL

# Create root RLM with max_depth=2
rlm = RLM_REPL(
    model="gpt-3.5-turbo",
    recursive_model="gpt-3.5-turbo",
    max_depth=2,  # Root can spawn sub-RLMs that can spawn their own sub-RLMs
    enable_logging=True
)

# Use as before
result = rlm.completion(context=large_context, query="Analyze this...")
```

### Depth Hierarchy Example
With `max_depth=3`:
- **Depth 0 (Root)**: Full REPL, can call sub-LLMs
- **Depth 1 (Sub)**: Full REPL, can call sub-LLMs
- **Depth 2 (Sub-sub)**: Full REPL, calls terminal Sub_RLM
- **Depth 3 (Terminal)**: No REPL, direct API calls only

## Performance and Cost Considerations

### Iteration Management
- Root LLM gets full `max_iterations`
- Sub-RLMs get reduced iterations: `max_iterations // (depth + 1)`
- Minimum of 3 iterations guaranteed for sub-RLMs
- Prevents excessive API costs at deeper levels

### Cost Implications
- **Higher depth = More API calls**: Each recursive level adds overhead
- **Use judiciously**: Multi-depth is powerful but expensive
- **Recommended**: Use `max_depth=2` or `max_depth=3` for most use cases
- **Default**: `max_depth=1` maintains original behavior (no extra cost)

## Backward Compatibility

✅ **Fully backward compatible**
- Default `max_depth=1` maintains original single-depth behavior
- All existing code works without changes
- No breaking changes to any APIs
- Sub-LLMs are still terminal by default

## Testing

Run the multi-depth example:
```bash
# Activate conda environment
conda activate min-verify

# Run multi-depth example
python extra/example_multi_depth.py
```

## Use Cases

### When to Use Multi-Depth
✅ **Complex hierarchical problems**: Break down into sub-problems that need their own code execution
✅ **Multi-step analysis**: Each step requires its own REPL environment
✅ **Divide-and-conquer**: Recursively decompose large tasks
✅ **Nested document processing**: Process documents that contain sub-documents requiring analysis

### When to Use Single Depth (max_depth=1)
✅ **Simple tasks**: Single level of sub-LLM calls is sufficient
✅ **Cost-sensitive**: Minimize API calls
✅ **Straightforward analysis**: No need for nested problem decomposition

## Technical Implementation Details

### Recursive Sub-RLM Creation
- Each `REPLEnv` checks `depth < max_depth - 1` to decide sub-RLM type
- Recursive sub-RLMs are created as fresh `RLM_REPL` instances
- Each instance has isolated state (messages, REPL environment, etc.)
- Depth is incremented automatically: `depth + 1`

### Parallel Batch Processing
- `_batch_recursive_completion()` handles parallel RLM_REPL calls
- Uses `ThreadPoolExecutor` for concurrent execution
- Each completion gets its own `RLM_REPL` instance to avoid state conflicts
- Results maintain input order

### Depth Tracking
- Depth is passed through constructor chain: `RLM_REPL` → `REPLEnv` → `RLM_REPL`
- Loggers include depth prefix: `[Depth {depth}]`
- Helps debug and trace recursive call chains

## Future Enhancements

Potential improvements for future versions:
- [ ] Depth-specific system prompts (customize behavior per depth)
- [ ] Cost tracking per depth level
- [ ] Dynamic depth adjustment based on problem complexity
- [ ] Depth-aware iteration limits (more sophisticated than current linear reduction)
- [ ] Visualization tools for recursive call trees

## Migration Guide

No migration needed! The changes are additive and fully backward compatible.

To use multi-depth recursion:
1. Set `max_depth` parameter when creating `RLM_REPL` (default is 1)
2. Existing code continues to work as-is
3. New recursive capabilities are automatically available

---

**Implementation Date**: January 2026  
**Implemented By**: AI Assistant  
**Status**: ✅ Complete and tested

