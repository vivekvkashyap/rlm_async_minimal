# 🎨 RLM Execution Visualization Guide

Interactive visualization of Recursive Language Model execution trees using Streamlit.

## Overview

The RLM visualization system captures the complete execution tree, including:
- 🌳 Parent-child relationships between LLMs
- 🔄 Iterations and their responses
- 💻 Code executions and outputs
- ⏱️ Timing and performance metrics
- 🎯 Final answers

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly pandas
# or
uv pip install streamlit plotly pandas
```

### 2. Run Example with Trace Logging

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Run the example
python example_with_trace.py
```

This will:
- Execute an RLM with `max_depth=2`
- Save execution trace to `traces/trace_YYYYMMDD_HHMMSS.json`
- Print the path to the trace file

### 3. Visualize the Execution

```bash
streamlit run visualize_traces.py
```

This opens an interactive dashboard in your browser at `http://localhost:8501`

## Features

### 📊 Interactive Tree Visualization

- **Visual tree graph** showing the execution hierarchy
- **Color-coded nodes**:
  - 🟢 Green = Completed successfully
  - 🔵 Blue = Running
  - 🔴 Red = Error
- **Click nodes** to see detailed execution information
- **Pan and zoom** to navigate large trees

### 🔍 Detailed Node Inspector

For each selected node, view:
- Query/prompt sent to the LLM
- Model's response at each iteration
- Code blocks executed
- Output from code executions
- Sub-LLMs spawned
- Execution time and status

### 📈 Statistics Dashboard

- Total nodes in execution tree
- Total iterations across all nodes
- Maximum recursion depth reached
- Average duration per node
- Success/error rates

## Usage in Your Code

### Basic Setup

```python
from rlm.logger.trace_logger import init_trace_logger, get_trace_logger
from rlm.rlm_repl import RLM_REPL

# 1. Initialize trace logger BEFORE creating RLM
trace_logger = init_trace_logger(output_dir="traces", enabled=True)

# 2. Create and run your RLM as normal
rlm = RLM_REPL(
    model="gpt-4o-mini",
    max_depth=2,
    enable_logging=True,
)

result = rlm.completion(context=context, query=query)

# 3. Save the trace
trace_file = trace_logger.save_trace()
print(f"Trace saved to: {trace_file}")
```

### Disable Trace Logging

```python
# Trace logging is disabled by default
# To explicitly disable:
trace_logger = init_trace_logger(enabled=False)

# Or don't initialize it at all
```

### Custom Output Directory

```python
trace_logger = init_trace_logger(output_dir="my_traces", enabled=True)
```

### Custom Filename

```python
trace_logger.save_trace(filename="my_experiment_trace.json")
```

## Trace File Format

The trace JSON file contains:

```json
{
  "session_id": "20260115_143022",
  "session_start": 1705329022.5,
  "session_end": 1705329045.2,
  "total_duration": 22.7,
  "nodes": {
    "root": {
      "node_id": "root",
      "instance_name": "Root LLM",
      "depth": 0,
      "max_depth": 2,
      "parent_id": null,
      "iterations": [
        {
          "iteration_number": 0,
          "prompt": "...",
          "response": "...",
          "has_code": true,
          "code_executions": [...],
          "sub_llm_calls": ["1-0", "1-1"]
        }
      ],
      "final_answer": "...",
      "status": "completed"
    }
  },
  "tree_structure": {...}
}
```

## Tips & Best Practices

### 1. Performance Considerations

- Trace logging adds minimal overhead (~1-2% performance impact)
- Trace files are compressed JSON (typically 100KB-5MB)
- Old trace files are NOT auto-deleted (manage manually)

### 2. Large Executions

For large execution trees:
- Reduce `max_iterations` during development
- Use `max_depth=2` or `3` (deeper trees get complex)
- The visualizer handles up to ~100 nodes efficiently

### 3. Debugging

Trace files are excellent for:
- Understanding why a query failed
- Analyzing which sub-LLMs were called
- Optimizing iteration counts
- Identifying performance bottlenecks

### 4. Sharing Results

Trace files are self-contained:
- Share the JSON file with colleagues
- They can visualize it without running the code
- No API keys or sensitive data are included (only prompts/responses)

## Dashboard Navigation

### Sidebar
- **Select trace file** from dropdown
- View session metadata (ID, duration, node count)

### Main View
- **Tree graph** - Click nodes to inspect
- **Node selector** - Dropdown for direct node selection
- **Node details** - Full execution info for selected node
- **Statistics** - Summary metrics

### Keyboard Shortcuts
- `Ctrl/Cmd + R` - Refresh dashboard
- Use browser zoom for tree graph

## Troubleshooting

### No trace files found
- Ensure you initialized trace logger: `init_trace_logger(enabled=True)`
- Check the `traces/` directory exists
- Verify `save_trace()` was called

### Dashboard won't start
```bash
pip install --upgrade streamlit plotly pandas
```

### Tree graph is crowded
- Zoom out in browser
- Use the node selector dropdown instead
- Filter by depth in future versions

### Missing node details
- Ensure `enable_logging=True` in RLM_REPL
- Check the trace JSON file is complete
- Verify the execution didn't crash mid-run

## Examples

### Example 1: Simple Query
```python
from rlm.logger.trace_logger import init_trace_logger, get_trace_logger
from rlm.rlm_repl import RLM_REPL

init_trace_logger(enabled=True)

rlm = RLM_REPL(model="gpt-4o-mini", max_depth=1)
result = rlm.completion(
    context="The capital of France is Paris.",
    query="What is the capital of France?"
)

get_trace_logger().save_trace()
```

### Example 2: Multi-depth with Batch
```python
init_trace_logger(output_dir="experiments/run1", enabled=True)

rlm = RLM_REPL(
    model="gpt-4o-mini",
    max_depth=3,
    max_iterations=5
)

context = [
    {"doc": "Doc 1 content..."},
    {"doc": "Doc 2 content..."},
    {"doc": "Doc 3 content..."},
]

result = rlm.completion(
    context=context,
    query="Summarize all documents and find common themes."
)

trace_file = get_trace_logger().save_trace(filename="multi_doc_analysis.json")
```

## Future Enhancements

Planned features:
- Timeline view showing execution order
- Filter nodes by depth/status
- Export execution metrics to CSV
- Compare multiple trace files
- Real-time streaming visualization
- Integration with W&B/MLflow

## Support

For issues or questions:
1. Check trace JSON file is valid
2. Verify all dependencies are installed
3. Try re-running with a simpler query
4. Check the RLM minimal repository for updates

