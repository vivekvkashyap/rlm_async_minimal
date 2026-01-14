# RLM Async Batch Processing Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                         │
│                    (main.py, example_batch.py)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ rlm.completion(context, query)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RLM_REPL Class                            │
│                     (rlm/rlm_repl.py)                            │
│                                                                   │
│  • Root LLM orchestration                                        │
│  • Iterative planning loop                                       │
│  • Message history management                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Generates code with llm_query/llm_batch
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        REPLEnv Class                             │
│                       (rlm/repl.py)                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              REPL Globals (Available Functions)           │  │
│  │                                                            │  │
│  │  • context: The input data                                │  │
│  │  • llm_query(prompt: str) -> str                          │  │
│  │  • llm_batch(prompts: List[str]) -> List[str]  ⚡NEW      │  │
│  │  • FINAL_VAR(var_name: str) -> str                        │  │
│  │  • print(), standard Python builtins                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  • Executes Python code in isolated environment                 │
│  • Manages state and variables                                  │
│  • Provides sub-LLM access                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Calls sub-LLM methods
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Sub_RLM Class                             │
│                       (rlm/repl.py)                              │
│                                                                   │
│  ┌─────────────────────┐     ┌──────────────────────────────┐  │
│  │  completion()       │     │  batch_completion()  ⚡NEW    │  │
│  │  (sequential)       │     │  (parallel)                   │  │
│  │                     │     │                               │  │
│  │  Single prompt      │     │  Multiple prompts             │  │
│  │  Returns 1 result   │     │  Returns N results            │  │
│  └──────────┬──────────┘     └──────────────┬───────────────┘  │
│             │                               │                   │
│             │                               │                   │
└─────────────┼───────────────────────────────┼───────────────────┘
              │                               │
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌─────────────────────────────────┐
│   OpenAIClient           │    │   AsyncOpenAIClient  ⚡NEW      │
│   (rlm/utils/llm.py)     │    │   (rlm/utils/llm.py)            │
│                          │    │                                 │
│  • Sync API calls        │    │  • Async API calls              │
│  • Single request        │    │  • Concurrent requests          │
│  • Traditional blocking  │    │  • asyncio.gather()             │
│                          │    │  • Rate limiting (semaphore)    │
└────────────┬─────────────┘    └────────────┬────────────────────┘
             │                               │
             │                               │
             ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OpenAI API                                │
│                                                                   │
│  Sequential:                   Parallel:                         │
│  Request 1 → Response 1        Request 1 ┐                      │
│  Request 2 → Response 2        Request 2 ├→ All responses       │
│  Request 3 → Response 3        Request 3 ┘                      │
│                                                                   │
│  Time: 3 × latency             Time: 1 × latency                │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow: Sequential vs Parallel

### Sequential Processing (OLD)

```
User Query
    ↓
RLM_REPL generates code:
    ```repl
    results = []
    for chunk in chunks:
        result = llm_query(f"Process {chunk}")
        results.append(result)
    ```
    ↓
REPL executes loop:
    ↓
    llm_query(chunk_1) → Sub_RLM.completion() → OpenAIClient → API (2s)
    ↓
    llm_query(chunk_2) → Sub_RLM.completion() → OpenAIClient → API (2s)
    ↓
    llm_query(chunk_3) → Sub_RLM.completion() → OpenAIClient → API (2s)
    ↓
Total: 6 seconds
```

### Parallel Processing (NEW)

```
User Query
    ↓
RLM_REPL generates code:
    ```repl
    prompts = [f"Process {chunk}" for chunk in chunks]
    results = llm_batch(prompts)
    ```
    ↓
REPL executes batch:
    ↓
    llm_batch([p1, p2, p3])
        ↓
    Sub_RLM.batch_completion([p1, p2, p3])
        ↓
    AsyncOpenAIClient.batch_completion_sync([p1, p2, p3])
        ↓
    asyncio.run(
        asyncio.gather(
            async_completion(p1),  ┐
            async_completion(p2),  ├→ All execute concurrently
            async_completion(p3)   ┘
        )
    )
        ↓
    [result1, result2, result3]
    ↓
Total: 2 seconds (3x faster!)
```

## Component Responsibilities

### 1. RLM_REPL (Root Orchestrator)
- **Input**: User query + context
- **Output**: Final answer
- **Responsibilities**:
  - Iterative planning and execution
  - Message history management
  - Calling root LLM for next actions
  - Detecting final answers

### 2. REPLEnv (Execution Environment)
- **Input**: Python code from root LLM
- **Output**: Execution results
- **Responsibilities**:
  - Safe code execution
  - State management (variables)
  - Exposing llm_query and llm_batch
  - Context loading and access

### 3. Sub_RLM (Sub-LLM Interface)
- **Input**: Prompts (single or batch)
- **Output**: LLM responses
- **Responsibilities**:
  - Sequential completion (existing)
  - Batch completion (new)
  - Error handling
  - Client management

### 4. OpenAIClient (Sync API Client)
- **Input**: Single prompt
- **Output**: Single response
- **Responsibilities**:
  - Synchronous API calls
  - Request formatting
  - Response parsing

### 5. AsyncOpenAIClient (Async API Client) ⚡NEW
- **Input**: Single or multiple prompts
- **Output**: Single or multiple responses
- **Responsibilities**:
  - Asynchronous API calls
  - Concurrent execution
  - Rate limiting (semaphore)
  - Event loop management
  - Error handling per request

## Async Implementation Details

### Event Loop Management

```python
# In AsyncOpenAIClient.batch_completion_sync()

try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Already in async context - use ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, self.batch_completion(...))
            return future.result()
    else:
        # No running loop - use asyncio.run
        return asyncio.run(self.batch_completion(...))
except RuntimeError:
    # No event loop - create one
    return asyncio.run(self.batch_completion(...))
```

### Rate Limiting with Semaphore

```python
# In AsyncOpenAIClient.batch_completion()

semaphore = asyncio.Semaphore(max_concurrent)

async def limited_completion(messages):
    async with semaphore:  # Only max_concurrent tasks run at once
        return await self.completion(messages)

tasks = [limited_completion(m) for m in messages_list]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Error Handling

```python
# Individual request failures don't crash entire batch
results = await asyncio.gather(*tasks, return_exceptions=True)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        processed_results.append(f"Error in batch request {i}: {str(result)}")
    else:
        processed_results.append(result)
```

## Performance Characteristics

### Time Complexity

**Sequential (llm_query in loop)**:
- Time = N × API_LATENCY
- Example: 10 chunks × 2s = 20s

**Parallel (llm_batch)**:
- Time = ⌈N / max_concurrent⌉ × API_LATENCY
- Example: 10 chunks / 10 concurrent = 1 batch × 2s = 2s
- 10x speedup!

### Space Complexity

**Memory Usage**:
- Sequential: O(1) - one request at a time
- Parallel: O(max_concurrent) - limited by semaphore

### Network Usage

**Bandwidth**:
- Same for both (same total data transferred)

**Connections**:
- Sequential: 1 connection reused
- Parallel: Up to max_concurrent connections

## Configuration Parameters

### max_concurrent (default: 10)

Controls how many API requests run simultaneously:

```python
# Conservative (avoid rate limits)
results = llm_batch(prompts, max_concurrent=3)

# Balanced (default)
results = llm_batch(prompts, max_concurrent=10)

# Aggressive (if you have high rate limits)
results = llm_batch(prompts, max_concurrent=20)
```

**Tradeoffs**:
- **Lower**: More conservative, less likely to hit rate limits
- **Higher**: Faster processing, but may hit rate limits

## Integration Points

### Where Batch Processing Helps Most

1. **Chunked Context Processing**
   - Split large context into chunks
   - Process each chunk in parallel
   - Combine results

2. **Multi-Section Analysis**
   - Document with multiple sections
   - Analyze each section independently
   - Aggregate findings

3. **Repeated Queries**
   - Same question on different data
   - Batch all queries together
   - Get all answers at once

4. **Map-Reduce Patterns**
   - Map: Process chunks in parallel (llm_batch)
   - Reduce: Combine results (llm_query)

## Monitoring and Debugging

### Logging

The system logs execution times:
```
[REPL] Executing code... (completed in 2.34s)
```

### Performance Metrics

Track these to measure improvement:
- Total execution time
- Number of sub-LLM calls
- Sequential vs parallel time
- Speedup ratio

### Debug Tips

1. **Check batch sizes**: Too small = overhead, too large = memory
2. **Monitor rate limits**: Adjust max_concurrent if hitting limits
3. **Verify parallelism**: Compare sequential vs parallel times
4. **Check error rates**: Individual failures in batch results

## Future Architecture Enhancements

### Potential Improvements

1. **Streaming Responses**
   ```python
   async for result in llm_batch_stream(prompts):
       process(result)  # Process as they arrive
   ```

2. **Caching Layer**
   ```python
   # Deduplicate identical prompts
   unique_prompts = set(prompts)
   cached_results = cache.get(unique_prompts)
   ```

3. **Full Async RLM**
   ```python
   # End-to-end async processing
   result = await rlm.completion_async(context, query)
   ```

4. **Adaptive Concurrency**
   ```python
   # Automatically adjust based on rate limits
   adaptive_batch(prompts, auto_tune=True)
   ```

---

**Last Updated**: January 2026  
**Version**: 1.0 with Async Batch Processing

