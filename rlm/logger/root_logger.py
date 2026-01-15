"""
Root (colorful) logger for RLM client that tracks model outputs and message changes.
Uses Rich library for beautiful output with thread-safe printing.

Supports buffered mode for collecting output that can be printed later in order.
"""

from typing import List, Dict, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich import box
from rich.rule import Rule
import threading
import sys
import io

# Global lock for thread-safe console printing
_print_lock = threading.Lock()

# Store the REAL stdout at module load time
_real_stdout = sys.__stdout__


def get_llm_style(depth: int, instance_name: str):
    """Get style configuration based on LLM type."""
    if depth == 0:
        return {
            'color': 'bright_green',
            'border': 'green',
            'box': box.DOUBLE,
            'emoji': '🟢',
            'char': '═',
        }
    elif 'Sub Root' in instance_name:
        return {
            'color': 'bright_cyan',
            'border': 'cyan',
            'box': box.ROUNDED,
            'emoji': '🔵',
            'char': '─',
        }
    else:
        return {
            'color': 'bright_magenta',
            'border': 'magenta',
            'box': box.SIMPLE,
            'emoji': '🟣',
            'char': '·',
        }


class ColorfulLogger:
    """
    A colorful logger using Rich library for beautiful, thread-safe output.
    
    Supports buffered mode: when buffered=True, output is collected and can be
    retrieved with get_buffer() for printing later in a specific order.
    """
    
    def __init__(
        self, 
        enabled: bool = True, 
        depth: int = 0,
        max_depth: int = 3,
        instance_name: str = "Root LLM",
        max_iterations: int = 20,
        buffered: bool = False,  # NEW: Buffer mode for ordered output
    ):
        self.enabled = enabled
        self.depth = depth
        self.max_depth = max_depth
        self.instance_name = instance_name
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.conversation_step = 0
        self.current_query = ""
        self.session_start_time = None
        
        # Buffered mode - collect output instead of printing immediately
        self.buffered = buffered
        self._output_buffer = io.StringIO() if buffered else None
        
        # Get style for this LLM type
        self.style = get_llm_style(depth, instance_name)
    
    def update_iteration(self, current_iteration: int):
        """Update the current iteration."""
        self.current_iteration = current_iteration
    
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
            # Collect in buffer for later
            self._output_buffer.write(output)
        else:
            # Print atomically to real stdout
            with _print_lock:
                _real_stdout.write(output)
                _real_stdout.flush()
    
    def log_query_start(self, query: str):
        """Log the start of a new query."""
        if not self.enabled:
            return
            
        self.current_query = query
        self.conversation_step = 0
        self.session_start_time = datetime.now()
        
        def render(console):
            header = Text()
            header.append(f"{self.style['emoji']} {self.instance_name}", style=f"bold {self.style['color']}")
            header.append(f" (depth {self.depth} of {self.max_depth})", style="dim")
            header.append(f" | {datetime.now().strftime('%H:%M:%S')}", style="dim italic")
            
            query_display = query[:300] + "..." if len(query) > 300 else query
            
            content = Text()
            content.append("📋 QUERY: ", style="bold")
            content.append(query_display)
            
            panel = Panel(
                content,
                title=header,
                title_align="left",
                border_style=self.style['border'],
                box=self.style['box'],
            )
            
            console.print()
            console.print(panel)
        
        self._output(render)
    
    def log_initial_messages(self, messages: List[Dict[str, str]]):
        """Log the initial messages setup."""
        if not self.enabled:
            return
        
        def render(console):
            console.print(Text(f"   📝 Initial Messages: {len(messages)} message(s) loaded", style="dim"))
        
        self._output(render)
    
    def log_iteration_start(self, iteration: int):
        """Log the start of a new iteration with a box."""
        if not self.enabled:
            return
        
        self.update_iteration(iteration)
        
        def render(console):
            iter_text = Text()
            iter_text.append(f"🔄 ITERATION {iteration}/{self.max_iterations}", style=f"bold {self.style['color']}")
            iter_text.append(f" [{self.instance_name}]", style="bold")
            iter_text.append(f" (depth {self.depth} of {self.max_depth})", style="dim")
            
            console.print()
            console.print(Rule(iter_text, style=self.style['border'], characters=self.style['char']))
        
        self._output(render)
    
    def log_model_response(self, response: str, has_tool_calls: bool):
        """Log the model's response."""
        if not self.enabled:
            return
            
        self.conversation_step += 1
        step = self.conversation_step
        
        def render(console):
            display_response = response[:400] + "..." if len(response) > 400 else response
            
            content = Text()
            content.append(display_response)
            
            if has_tool_calls:
                subtitle = Text("⚡ Contains tool calls - will execute", style="bold yellow")
            else:
                subtitle = Text("✓ No tool calls - processing complete", style="bold green")
            
            panel = Panel(
                content,
                title=f"[bold]💬 Response (Step {step})[/bold]",
                subtitle=subtitle,
                border_style="white",
                box=box.ROUNDED,
            )
            
            console.print(panel)
        
        self._output(render)
    
    def log_tool_execution(self, tool_call_str: str, tool_result: str):
        """Log tool execution and result."""
        if not self.enabled:
            return
        
        def render(console):
            display_result = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
            
            content = Text()
            content.append("🔧 Tool: ", style="bold yellow")
            content.append(f"{tool_call_str}\n", style="yellow")
            content.append("📤 Result: ", style="bold green")
            content.append(display_result)
            
            console.print(Panel(content, title="[bold cyan]⚙️ Tool Execution[/bold cyan]", border_style="cyan", box=box.ROUNDED))
        
        self._output(render)
    
    def log_final_response(self, response: str):
        """Log the final response with a prominent box."""
        if not self.enabled:
            return
        
        def render(console):
            display = response[:2000] + "..." if len(response) > 2000 else response
            
            panel = Panel(
                Text(display),
                title=f"[bold white on green] ✅ FINAL ANSWER [{self.instance_name}] [/bold white on green]",
                border_style="bold green",
                box=box.DOUBLE,
                padding=(1, 2),
            )
            
            console.print()
            console.print(Rule(style="green", characters="═"))
            console.print(panel)
            console.print(Rule(style="green", characters="═"))
            console.print()
        
        self._output(render)
    
    def log_iteration_end(self):
        """Log the end of an iteration."""
        if not self.enabled:
            return
        
        def render(console):
            console.print(Rule(style="dim", characters="─"))
        
        self._output(render)


def flush_buffer_to_stdout(buffer_content: str):
    """Flush buffered content to stdout atomically."""
    with _print_lock:
        _real_stdout.write(buffer_content)
        _real_stdout.flush()
