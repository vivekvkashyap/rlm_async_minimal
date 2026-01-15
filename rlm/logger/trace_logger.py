"""
Execution Trace Logger for RLM.

Captures the execution tree structure, prompts, responses, code executions,
and parent-child relationships for visualization.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading


@dataclass
class CodeExecution:
    """Represents a single code execution."""
    code: str
    stdout: str
    stderr: str
    execution_time: float
    execution_number: int


@dataclass
class LLMNode:
    """Represents a single LLM instance in the execution tree."""
    node_id: str  # Unique identifier (e.g., "root", "1-0", "1-0.a")
    instance_name: str  # Display name (e.g., "Root LLM", "Sub Root LLM 1-0")
    depth: int
    max_depth: int
    parent_id: Optional[str] = None
    
    # Execution details
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Results
    final_answer: Optional[str] = None
    status: str = "running"  # running, completed, error
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_id": self.node_id,
            "instance_name": self.instance_name,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time else None,
            "iterations": self.iterations,
            "final_answer": self.final_answer,
            "status": self.status,
        }


@dataclass
class IterationData:
    """Represents a single iteration within an LLM node."""
    iteration_number: int
    prompt: str
    response: str
    has_code: bool
    code_executions: List[Dict[str, Any]] = field(default_factory=list)
    sub_llm_calls: List[str] = field(default_factory=list)  # IDs of spawned sub-LLMs
    timestamp: float = field(default_factory=time.time)


class ExecutionTraceLogger:
    """
    Logs the execution trace of RLM for visualization.
    Thread-safe and can be used across parallel executions.
    """
    
    def __init__(self, output_dir: str = "traces", enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Execution tree: maps node_id -> LLMNode
        self.nodes: Dict[str, LLMNode] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()
        
    def create_node(
        self,
        node_id: str,
        instance_name: str,
        depth: int,
        max_depth: int,
        parent_id: Optional[str] = None
    ) -> None:
        """Create a new node in the execution tree."""
        if not self.enabled:
            return
        
        with self._lock:
            node = LLMNode(
                node_id=node_id,
                instance_name=instance_name,
                depth=depth,
                max_depth=max_depth,
                parent_id=parent_id,
                start_time=time.time(),
            )
            self.nodes[node_id] = node
    
    def log_iteration_start(
        self,
        node_id: str,
        iteration_number: int,
        prompt: str
    ) -> None:
        """Log the start of an iteration."""
        if not self.enabled or node_id not in self.nodes:
            return
        
        with self._lock:
            iteration = {
                "iteration_number": iteration_number,
                "prompt": prompt,
                "response": "",
                "has_code": False,
                "code_executions": [],
                "sub_llm_calls": [],
                "timestamp": time.time(),
            }
            self.nodes[node_id].iterations.append(iteration)
    
    def log_response(
        self,
        node_id: str,
        iteration_number: int,
        response: str,
        has_code: bool
    ) -> None:
        """Log the model's response for an iteration."""
        if not self.enabled or node_id not in self.nodes:
            return
        
        with self._lock:
            node = self.nodes[node_id]
            if iteration_number < len(node.iterations):
                node.iterations[iteration_number]["response"] = response
                node.iterations[iteration_number]["has_code"] = has_code
    
    def log_code_execution(
        self,
        node_id: str,
        iteration_number: int,
        code: str,
        stdout: str,
        stderr: str,
        execution_time: float,
        execution_number: int
    ) -> None:
        """Log a code execution within an iteration."""
        if not self.enabled or node_id not in self.nodes:
            return
        
        with self._lock:
            node = self.nodes[node_id]
            if iteration_number < len(node.iterations):
                code_exec = {
                    "code": code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "execution_time": execution_time,
                    "execution_number": execution_number,
                }
                node.iterations[iteration_number]["code_executions"].append(code_exec)
    
    def log_sub_llm_spawn(
        self,
        parent_node_id: str,
        parent_iteration: int,
        child_node_id: str
    ) -> None:
        """Log when a parent spawns a child sub-LLM."""
        if not self.enabled or parent_node_id not in self.nodes:
            return
        
        with self._lock:
            parent = self.nodes[parent_node_id]
            if parent_iteration < len(parent.iterations):
                parent.iterations[parent_iteration]["sub_llm_calls"].append(child_node_id)
    
    def log_final_answer(
        self,
        node_id: str,
        final_answer: str
    ) -> None:
        """Log the final answer from a node."""
        if not self.enabled or node_id not in self.nodes:
            return
        
        with self._lock:
            self.nodes[node_id].final_answer = final_answer
            self.nodes[node_id].status = "completed"
            self.nodes[node_id].end_time = time.time()
    
    def log_error(
        self,
        node_id: str,
        error: str
    ) -> None:
        """Log an error for a node."""
        if not self.enabled or node_id not in self.nodes:
            return
        
        with self._lock:
            self.nodes[node_id].status = "error"
            self.nodes[node_id].final_answer = f"ERROR: {error}"
            self.nodes[node_id].end_time = time.time()
    
    def save_trace(self, filename: Optional[str] = None) -> Path:
        """Save the execution trace to a JSON file."""
        if not self.enabled:
            return None
        
        if filename is None:
            filename = f"trace_{self.session_id}.json"
        
        output_path = self.output_dir / filename
        
        with self._lock:
            trace_data = {
                "session_id": self.session_id,
                "session_start": self.session_start,
                "session_end": time.time(),
                "total_duration": time.time() - self.session_start,
                "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                "tree_structure": self._build_tree_structure(),
            }
        
        with open(output_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        print(f"📊 Execution trace saved to: {output_path}")
        return output_path
    
    def _build_tree_structure(self) -> Dict:
        """Build a hierarchical tree structure for visualization."""
        # Find root nodes (no parent)
        root_nodes = [node_id for node_id, node in self.nodes.items() if node.parent_id is None]
        
        def build_subtree(node_id: str) -> Dict:
            """Recursively build subtree."""
            node = self.nodes[node_id]
            children_ids = [nid for nid, n in self.nodes.items() if n.parent_id == node_id]
            
            return {
                "id": node_id,
                "name": node.instance_name,
                "depth": node.depth,
                "status": node.status,
                "children": [build_subtree(child_id) for child_id in children_ids]
            }
        
        return {
            "roots": [build_subtree(root_id) for root_id in root_nodes]
        }


# Global trace logger instance
_global_trace_logger: Optional[ExecutionTraceLogger] = None


def get_trace_logger() -> Optional[ExecutionTraceLogger]:
    """Get the global trace logger instance."""
    return _global_trace_logger


def init_trace_logger(output_dir: str = "traces", enabled: bool = True) -> ExecutionTraceLogger:
    """Initialize the global trace logger."""
    global _global_trace_logger
    _global_trace_logger = ExecutionTraceLogger(output_dir=output_dir, enabled=enabled)
    return _global_trace_logger


def reset_trace_logger() -> None:
    """Reset the global trace logger."""
    global _global_trace_logger
    _global_trace_logger = None

