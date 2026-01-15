"""
REPL Environment Logger for code executions.
Uses Rich library for beautiful Jupyter-like output with thread-safe printing.

Supports buffered mode for collecting output that can be printed later in order.
"""

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.rule import Rule
from dataclasses import dataclass
from typing import List, Optional
import threading
import sys
import io

# Global lock for thread-safe console printing
_print_lock = threading.Lock()

# Store the REAL stdout at module load time
_real_stdout = sys.__stdout__


@dataclass
class CodeExecution:
    """Represents a single code execution in the REPL."""
    code: str
    stdout: str
    stderr: str
    execution_number: int
    execution_time: Optional[float] = None


def get_depth_style(depth: int):
    """Get style based on depth."""
    styles = {
        0: {'color': 'bright_green', 'border': 'green', 'box': box.DOUBLE, 'emoji': '🟢'},
        1: {'color': 'bright_cyan', 'border': 'cyan', 'box': box.ROUNDED, 'emoji': '🔵'},
        2: {'color': 'bright_magenta', 'border': 'magenta', 'box': box.SIMPLE, 'emoji': '🟣'},
    }
    return styles.get(depth, styles[2])


class REPLEnvLogger:
    """
    Logger for REPL environment code executions.
    Uses Rich library for beautiful Jupyter-like output.
    
    Supports buffered mode: when buffered=True, output is collected and can be
    retrieved with get_buffer() for printing later in a specific order.
    """
    
    def __init__(
        self, 
        max_output_length: int = 1500, 
        enabled: bool = True, 
        depth: int = 0,
        instance_name: str = "Root LLM",
        buffered: bool = False,  # NEW: Buffer mode for ordered output
    ):
        self.enabled = enabled
        self.depth = depth
        self.instance_name = instance_name
        self.executions: List[CodeExecution] = []
        self.execution_count = 0
        self.max_output_length = max_output_length
        
        # Buffered mode
        self.buffered = buffered
        self._output_buffer = io.StringIO() if buffered else None
        
        # Get style for this depth
        self.style = get_depth_style(depth)
    
    def get_buffer(self) -> str:
        """Get the buffered output (only in buffered mode)."""
        if self._output_buffer:
            return self._output_buffer.getvalue()
        return ""
    
    def clear_buffer(self):
        """Clear the output buffer."""
        if self._output_buffer:
            self._output_buffer = io.StringIO()
    
    def _render_to_string(self, render_func) -> str:
        """Render to a string buffer."""
        buffer = io.StringIO()
        buffer_console = Console(file=buffer, force_terminal=True, width=120)
        render_func(buffer_console)
        return buffer.getvalue()
    
    def _output(self, render_func):
        """Output rendered content - either buffer it or print atomically."""
        output = self._render_to_string(render_func)
        
        if self.buffered:
            self._output_buffer.write(output)
        else:
            with _print_lock:
                _real_stdout.write(output)
                _real_stdout.flush()
    
    def _truncate_text(self, text: str, max_len: int = None) -> str:
        """Truncate text, showing first and last parts."""
        if max_len is None:
            max_len = self.max_output_length
        
        if len(text) <= max_len:
            return text
        
        first_len = int(max_len * 0.4)
        last_len = int(max_len * 0.4)
        truncated_count = len(text) - first_len - last_len
        
        return f"{text[:first_len]}\n\n... [{truncated_count} characters truncated] ...\n\n{text[-last_len:]}"
    
    def log_execution(self, code: str, stdout: str, stderr: str = "", execution_time: Optional[float] = None) -> None:
        """Log a code execution."""
        self.execution_count += 1
        execution = CodeExecution(
            code=code,
            stdout=stdout,
            stderr=stderr,
            execution_number=self.execution_count,
            execution_time=execution_time
        )
        self.executions.append(execution)
    
    def display_last(self) -> None:
        """Display the last logged execution with Rich formatting."""
        if not self.enabled or not self.executions:
            return
        
        exec_data = self.executions[-1]
        style = self.style
        instance_name = self.instance_name
        depth = self.depth
        
        def render(console):
            # ═══════ CODE INPUT PANEL ═══════
            syntax = Syntax(
                exec_data.code,
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )
            
            input_title = Text()
            input_title.append(f"{style['emoji']} ", style="bold")
            input_title.append(f"In [{exec_data.execution_number}]", style=f"bold {style['color']}")
            input_title.append(f" [{instance_name}]", style="bold")
            input_title.append(f" (depth {depth})", style="dim")
            
            input_panel = Panel(
                syntax,
                title=input_title,
                title_align="left",
                border_style=style['border'],
                box=style['box'],
            )
            console.print(input_panel)
            
            # ═══════ OUTPUT PANEL ═══════
            if exec_data.stderr:
                error_text = self._truncate_text(exec_data.stderr)
                
                output_title = Text()
                output_title.append("❌ ", style="bold")
                output_title.append(f"Error [{exec_data.execution_number}]", style="bold red")
                output_title.append(f" [{instance_name}]", style="dim")
                
                output_panel = Panel(
                    Text(error_text, style="red"),
                    title=output_title,
                    title_align="left",
                    border_style="red",
                    box=box.ROUNDED,
                )
                console.print(output_panel)
                
            elif exec_data.stdout:
                output_text = self._truncate_text(exec_data.stdout)
                
                time_str = f" ⏱️ {exec_data.execution_time:.4f}s" if exec_data.execution_time else ""
                
                output_title = Text()
                output_title.append("📤 ", style="bold")
                output_title.append(f"Out [{exec_data.execution_number}]", style=f"bold green")
                output_title.append(f" [{instance_name}]", style="dim")
                output_title.append(time_str, style="dim italic")
                
                output_panel = Panel(
                    Text(output_text),
                    title=output_title,
                    title_align="left",
                    border_style="green",
                    box=box.ROUNDED,
                )
                console.print(output_panel)
                
            else:
                time_str = f" ⏱️ {exec_data.execution_time:.4f}s" if exec_data.execution_time else ""
                console.print(f"   📤 Out [{exec_data.execution_number}]: (no output){time_str}", style="dim")
            
            console.print()
        
        self._output(render)
    
    def display_all(self) -> None:
        """Display all logged executions."""
        if not self.enabled:
            return
        
        original_executions = self.executions.copy()
        for i, exec_item in enumerate(original_executions):
            self.executions = [exec_item]
            self.display_last()
            if i < len(original_executions) - 1:
                def render_sep(console):
                    console.print(Rule(style="dim", characters="─"))
                self._output(render_sep)
        self.executions = original_executions
    
    def clear(self) -> None:
        """Clear all logged executions."""
        self.executions.clear()
        self.execution_count = 0
