# Extra Examples and Tests

This folder contains example scripts and test files for the RLM implementation.

## Example Scripts

- **`example_batch.py`** - Demonstrates async batch processing with `llm_batch()`
- **`example_ollama.py`** - Needle-in-haystack example with large context
- **`example_parallel_chunks.py`** - Parallel processing of multiple document chunks
- **`example_semantic.py`** - Multi-document semantic analysis using parallel sub-LLMs

## Test Scripts

- **`test_batch.py`** - Tests for batch processing functionality
- **`test_parallel_simple.py`** - Simple test to verify parallel execution
- **`test_parallel_speedup.py`** - Comprehensive test comparing sequential vs parallel performance

## Other Files

- **`new_extra_notebook.ipynb`** - Jupyter notebook with additional examples

## Running Examples

All examples require `OPENAI_API_KEY` to be set:

```bash
export OPENAI_API_KEY=your-key-here
python extra/example_batch.py
python extra/example_semantic.py
```

## Running Tests

```bash
python extra/test_parallel_speedup.py
python extra/test_parallel_simple.py
```

