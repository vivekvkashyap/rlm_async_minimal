import sys
import io
import threading
import json
import tempfile
import os
import time
import asyncio
import concurrent.futures
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

from rlm import RLM

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from rlm.rlm_repl import RLM_REPL


# Simple sub LM for REPL environment - TERMINAL node (no REPL, no further recursion)
class Sub_RLM(RLM):
    """
    Terminal LLM client for REPL environment.
    This is a simple API wrapper without REPL capabilities.
    Used as the final depth level where no further recursion is allowed.
    """
    
    def __init__(self, model: str = "gpt-5", api_key: Optional[str] = None):
        # Configuration - model can be specified
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model = model

        # Initialize OpenAI clients (sync and async)
        from rlm.utils.llm import OpenAIClient, AsyncOpenAIClient
        self.client = OpenAIClient(api_key=self.api_key, model=model)
        self.async_client = AsyncOpenAIClient(api_key=self.api_key, model=model)
        
    
    def completion(self, prompt) -> str:
        """
        Simple LM query for sub-LM call.
        """
        try:
            # Handle both string and dictionary/list inputs
            response = self.client.completion(
                messages=prompt,
                timeout=300
            )
            
            return response
                
        except Exception as e:
            error_msg = f"Error making LLM query: {str(e)}"
            return error_msg
    
    def batch_completion(self, prompts: List[str], max_concurrent: int = 10) -> List[str]:
        """
        Batch LM query for parallel sub-LM calls.
        
        Args:
            prompts: List of prompts to send to the LLM
            max_concurrent: Maximum number of concurrent requests (default: 10)
            
        Returns:
            List of responses in the same order as input prompts
        """
        try:
            # Use the async client's synchronous wrapper for batch completion
            responses = self.async_client.batch_completion_sync(
                messages_list=prompts,
                max_concurrent=max_concurrent,
                timeout=300
            )
            return responses
                
        except Exception as e:
            error_msg = f"Error making batch LLM query: {str(e)}"
            # Return error for all prompts
            return [error_msg] * len(prompts)
    
    def cost_summary(self) -> dict[str, float]:
        raise NotImplementedError("Cost tracking is not implemented for the Sub-RLM.")
    
    def reset(self):
        raise NotImplementedError("Reset is not implemented for the Sub-RLM.")


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float

    def __init__(self, stdout: str, stderr: str, locals: dict, execution_time: float=None):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
    
    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={self.locals}, execution_time={self.execution_time})"

class REPLEnv:
    """
    REPL Environment for executing Python code with LLM query capabilities.
    
    Supports multi-depth recursion:
    - If depth < max_depth - 1: sub_rlm is an RLM_REPL (has its own REPL)
    - If depth >= max_depth - 1: sub_rlm is a Sub_RLM (terminal, no REPL)
    """
    
    def __init__(
        self,
        recursive_model: str = "gpt-5-mini",
        context_json: Optional[dict | list] = None,
        context_str: Optional[str] = None,
        setup_code: str = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 20,
        api_key: Optional[str] = None,
        enable_logging: bool = False,
    ):
        # Store depth info
        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations_per_depth = max_iterations
        self.api_key = api_key
        self.enable_logging = enable_logging
        self.recursive_model = recursive_model
        
        # Store the original working directory
        self.original_cwd = os.getcwd()
        
        # Create temporary directory (but don't change global working directory)
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_d{depth}_")

        # Initialize sub_rlm based on depth
        # If we can go deeper (depth < max_depth - 1), create RLM_REPL with REPL capabilities
        # Otherwise, create terminal Sub_RLM (simple API call, no REPL)
        if depth < max_depth - 1:
            # Sub-RLM gets its own REPL environment and can recurse further
            # Import here to avoid circular imports
            from rlm.rlm_repl import RLM_REPL
            self.sub_rlm: RLM = RLM_REPL(
                api_key=api_key,
                model=recursive_model,
                recursive_model=recursive_model,
                depth=depth + 1,
                max_depth=max_depth,
                max_iterations=max_iterations,  # Pass parent's max_iterations so sub-RLM can calculate reduced value
                enable_logging=enable_logging,
            )
            self._sub_rlm_is_recursive = True
            print(f"[Depth {depth}] Created recursive sub_rlm (RLM_REPL at depth {depth + 1})")
        else:
            # Terminal node - simple API call, no REPL
            self.sub_rlm: RLM = Sub_RLM(model=recursive_model, api_key=api_key)
            self._sub_rlm_is_recursive = False
            print(f"[Depth {depth}] Created terminal sub_rlm (Sub_RLM - no REPL)")
        
        # Create safe globals with only string-safe built-ins
        self.globals = {
            '__builtins__': {
                # Safe built-ins for string manipulation
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
                'type': type, 'isinstance': isinstance, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct,
                'repr': repr, 'ascii': ascii, 'format': format,
                '__import__': __import__,  # Allow imports
                'open': open,  # Allow file access
                
                # Add commonly used built-ins that were missing
                'any': any, 'all': all, 'hasattr': hasattr, 'getattr': getattr,
                'setattr': setattr, 'delattr': delattr, 'dir': dir, 'vars': vars,
                'range': range,  # Add range function
                'reversed': reversed,  # Add reversed function
                'slice': slice,  # Add slice function
                'iter': iter,  # Add iter function
                'next': next,  # Add next function
                'pow': pow,  # Add pow function
                'divmod': divmod,  # Add divmod function
                'complex': complex,  # Add complex function
                'bytes': bytes,  # Add bytes function
                'bytearray': bytearray,  # Add bytearray function
                'memoryview': memoryview,  # Add memoryview function
                'hash': hash,  # Add hash function
                'id': id,  # Add id function
                'callable': callable,  # Add callable function
                'issubclass': issubclass,  # Add issubclass function
                'super': super,  # Add super function
                'property': property,  # Add property function
                'staticmethod': staticmethod,  # Add staticmethod function
                'classmethod': classmethod,  # Add classmethod function
                'object': object,  # Add object class
                'BaseException': BaseException,  # Add BaseException class
                'ArithmeticError': ArithmeticError,  # Add ArithmeticError class
                'LookupError': LookupError,  # Add LookupError class
                'EnvironmentError': EnvironmentError,  # Add EnvironmentError class
                'AssertionError': AssertionError,  # Add AssertionError class
                'NotImplementedError': NotImplementedError,  # Add NotImplementedError class
                'UnicodeError': UnicodeError,  # Add UnicodeError class
                'Warning': Warning,  # Add Warning class
                'UserWarning': UserWarning,  # Add UserWarning class
                'DeprecationWarning': DeprecationWarning,  # Add DeprecationWarning class
                'PendingDeprecationWarning': PendingDeprecationWarning,  # Add PendingDeprecationWarning class
                'SyntaxWarning': SyntaxWarning,  # Add SyntaxWarning class
                'RuntimeWarning': RuntimeWarning,  # Add RuntimeWarning class
                'FutureWarning': FutureWarning,  # Add FutureWarning class
                'ImportWarning': ImportWarning,  # Add ImportWarning class
                'UnicodeWarning': UnicodeWarning,  # Add UnicodeWarning class
                'BytesWarning': BytesWarning,  # Add BytesWarning class
                'ResourceWarning': ResourceWarning,  # Add ResourceWarning class
                
                # Add exception classes
                'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                'KeyError': KeyError, 'IndexError': IndexError, 'AttributeError': AttributeError,
                'FileNotFoundError': FileNotFoundError, 'OSError': OSError, 'IOError': IOError,
                'RuntimeError': RuntimeError, 'NameError': NameError, 'ImportError': ImportError,
                'StopIteration': StopIteration, 'GeneratorExit': GeneratorExit,
                'SystemExit': SystemExit, 'KeyboardInterrupt': KeyboardInterrupt,

                # Disallow the following built-ins
                'input': None,  # Block input
                'eval': None,  # Block eval
                'exec': None,  # Block exec
                'compile': None,  # Block compile
                'globals': None,  # Block globals access
                'locals': None,  # Block locals access
            }
        }
        self.locals = {}
        self._lock = threading.Lock()
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()

        self.load_context(context_json, context_str)
        
        def llm_query(prompt: str) -> str:
            """
            Query the LLM with the given prompt.
            
            If sub_rlm is RLM_REPL (recursive), the prompt is used as both context and query.
            If sub_rlm is Sub_RLM (terminal), the prompt is sent directly to the API.
            """
            if self._sub_rlm_is_recursive:
                # RLM_REPL.completion(context, query) - use prompt as context, extract query
                # The sub-RLM will use its own REPL environment to process this
                return self.sub_rlm.completion(context=prompt, query="Process this and provide your answer.")
            else:
                # Sub_RLM.completion(prompt) - direct API call
                return self.sub_rlm.completion(prompt)
        
        def llm_batch(prompts: List[str], max_concurrent: int = 10) -> List[str]:
            """
            Query the LLM with multiple prompts in parallel.
            
            Args:
                prompts: List of prompts to send to the LLM
                max_concurrent: Maximum number of concurrent requests (default: 10)
                
            Returns:
                List of responses in the same order as input prompts
                
            Example:
                prompts = [f"Summarize: {chunk}" for chunk in chunks]
                results = llm_batch(prompts)
            """
            if self._sub_rlm_is_recursive:
                # For recursive RLM_REPL, we need to run completions in parallel
                # Each completion uses its own REPL environment
                return self._batch_recursive_completion(prompts, max_concurrent)
            else:
                # For terminal Sub_RLM, use the async batch completion
                return self.sub_rlm.batch_completion(prompts, max_concurrent=max_concurrent)
        
        # Add (R)LM query functions to globals
        self.globals['llm_query'] = llm_query
        self.globals['llm_batch'] = llm_batch
        
        # Add FINAL_VAR function to globals
        def final_var(variable_name: str) -> str:
            """
            Return the value of a variable from the REPL environment as a final answer.
            This function is used by the model to return variables as final answers.
            """
            # Strip spaces, quotes, and newlines from variable name
            variable_name = variable_name.strip().strip('"').strip("'").strip('\n').strip('\r')
            try:
                # Check if variable exists in locals
                if variable_name in self.locals:
                    value = self.locals[variable_name]
                    return str(value)
                else:
                    return f"Error: Variable '{variable_name}' not found in REPL environment"
            except Exception as e:
                return f"Error retrieving variable '{variable_name}': {str(e)}"
        
        self.globals['FINAL_VAR'] = final_var
        
        # Finally, run any setup code if provided
        if setup_code:
            self.code_execution(setup_code)
    
    def _batch_recursive_completion(self, prompts: List[str], max_concurrent: int = 10) -> List[str]:
        """
        Run multiple RLM_REPL completions in parallel using ThreadPoolExecutor.
        
        Each completion creates its own RLM_REPL instance to avoid state conflicts.
        """
        from rlm.rlm_repl import RLM_REPL
        
        def run_single_completion(prompt: str) -> str:
            """Run a single RLM_REPL completion in a thread."""
            try:
                # Create a fresh RLM_REPL instance for this completion
                rlm = RLM_REPL(
                    api_key=self.api_key,
                    model=self.recursive_model,
                    recursive_model=self.recursive_model,
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    max_iterations=self.max_iterations_per_depth,  # Pass parent's max_iterations
                    enable_logging=self.enable_logging,
                )
                return rlm.completion(context=prompt, query="Process this and provide your answer.")
            except Exception as e:
                return f"Error in recursive completion: {str(e)}"
        
        # Run completions in parallel using ThreadPoolExecutor
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_idx = {
                executor.submit(run_single_completion, prompt): i 
                for i, prompt in enumerate(prompts)
            }
            
            # Collect results in order
            results = [None] * len(prompts)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Error: {str(e)}"
        
        return results
    
    def load_context(self, context_json: Optional[dict | list] = None, context_str: Optional[str] = None):
        # Write context JSON to temporary directory using absolute (temp dir) path
        if context_json is not None:
            context_path = os.path.join(self.temp_dir, "context.json")
            with open(context_path, "w") as f:
                json.dump(context_json, f, indent=2)
            context_code = (
                f"import json\n"
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = json.load(f)\n"
            )
            self.code_execution(context_code)
        
        if context_str is not None:
            context_path = os.path.join(self.temp_dir, "context.txt")
            with open(context_path, "w") as f:
                f.write(context_str)
            context_code = (
                f"import os\n"
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = f.read()\n"
            )
            self.code_execution(context_code)
    
    def __del__(self):
        """Clean up temporary directory when object is destroyed"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass 
    
    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr"""
        with self._lock:
            # Store original streams
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Create new buffers for this execution
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            try:
                # Redirect streams
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                yield stdout_buffer, stderr_buffer
            finally:
                # Restore original streams
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    
    @contextmanager
    def _temp_working_directory(self):
        """Context manager to temporarily change working directory for REPL execution"""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)
    
    def code_execution(self, code) -> REPLResult:
        """
        Simple code execution "notebook-style" in a REPL environment.
        """
        start_time = time.time()
        with self._capture_output() as (stdout_buffer, stderr_buffer):
            with self._temp_working_directory():
                try:
                    # Split code into import statements and other code
                    lines = code.split('\n')
                    import_lines = []
                    other_lines = []
                    
                    for line in lines:
                        if line.startswith(('import ', 'from ')) and not line.startswith('#'):
                            import_lines.append(line)
                        else:
                            other_lines.append(line)
                    
                    # Execute imports first in globals to make them available
                    if import_lines:
                        import_code = '\n'.join(import_lines)
                        exec(import_code, self.globals, self.globals)
                    
                    # Execute the rest of the code. We also want to print last expressions
                    if other_lines:
                        other_code = '\n'.join(other_lines)
                        # Create a combined namespace that includes both globals and locals
                        combined_namespace = {**self.globals, **self.locals}
                        
                        # Check if the last non-comment line is an expression
                        non_comment_lines = [line for line in other_lines if line and not line.startswith('#')]
                        
                        if non_comment_lines:
                            last_line = non_comment_lines[-1]
                            
                            # Check if the last line looks like an expression (not a statement)
                            is_expression = (
                                not last_line.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ', 'return ', 'yield ', 'break', 'continue', 'pass')) and
                                '=' not in last_line.split('#')[0] and  # Not an assignment
                                not last_line.endswith(':') and  # Not a control structure
                                not last_line.startswith('print(')  # Not an explicit print
                            )
                            
                            if is_expression:
                                try:
                                    # Execute all lines except the last one as statements
                                    if len(non_comment_lines) > 1:
                                        # Find where the last line starts in the original code
                                        last_line_start = -1
                                        for i, line in enumerate(other_lines):
                                            if line == last_line:
                                                last_line_start = i
                                                break
                                        
                                        if last_line_start > 0:
                                            statements_code = '\n'.join(other_lines[:last_line_start])
                                            exec(statements_code, combined_namespace, combined_namespace)
                                    
                                    # Evaluate the last line as an expression and print the result
                                    result = eval(last_line, combined_namespace, combined_namespace)
                                    if result is not None:
                                        print(repr(result))
                                        
                                except:
                                    # If evaluation fails, fall back to normal execution
                                    exec(other_code, combined_namespace, combined_namespace)
                            else:
                                # Execute normally as statements
                                exec(other_code, combined_namespace, combined_namespace)
                        else:
                            # Only comments, execute normally (though it won't do anything)
                            exec(other_code, combined_namespace, combined_namespace)
                        
                        # Update locals with any new variables created
                        for key, value in combined_namespace.items():
                            if key not in self.globals:
                                self.locals[key] = value
                    
                    stdout_content = stdout_buffer.getvalue()
                    stderr_content = stderr_buffer.getvalue()
                except Exception as e:
                    stderr_content = stderr_buffer.getvalue() + str(e)
                    stdout_content = stdout_buffer.getvalue()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store output in locals for access
        self.locals['_stdout'] = stdout_content
        self.locals['_stderr'] = stderr_content
        
        return REPLResult(stdout_content, stderr_content, self.locals.copy(), execution_time)
    
    def get_cost_summary(self):
        raise NotImplementedError("Cost tracking is not implemented for the REPL Environment.")
