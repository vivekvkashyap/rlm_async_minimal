"""
Simple Recursive Language Model (RLM) with REPL environment.

Supports multi-depth recursion where sub-LLMs can also have their own REPL
environments and spawn further sub-LLMs, up to a configurable max_depth.
"""

from typing import Dict, List, Optional, Any 

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils

from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_REPL(RLM):
    """
    LLM Client that can handle long contexts by recursively calling itself.
    
    Supports multi-depth recursion:
    - depth=0 is the root LLM
    - Each sub-LLM at depth < max_depth-1 gets its own REPL environment
    - Sub-LLMs at depth >= max_depth-1 are terminal (simple API calls)
    
    Example with max_depth=2:
        Root (depth=0) -> Sub-RLM with REPL (depth=1) -> Terminal Sub_RLM (no REPL)
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-5",
                 recursive_model: str = "gpt-5",
                 max_iterations: int = 20,
                 depth: int = 0,
                 max_depth: int = 1,
                 enable_logging: bool = False,
                 ):
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.llm = OpenAIClient(api_key, model)
        
        # Track recursive call depth to prevent infinite loops
        self.repl_env = None
        self.depth = depth
        self.max_depth = max_depth
        
        # Adjust iterations based on depth (deeper = fewer iterations to save cost)
        # Root gets full iterations, sub-levels get progressively fewer
        if depth == 0:
            self._max_iterations = max_iterations
        else:
            # Reduce iterations for deeper levels: max_iterations // (depth + 1)
            self._max_iterations = max(3, max_iterations // (depth + 1))
        
        # Initialize colorful logger with depth prefix
        self.logger = ColorfulLogger(enabled=enable_logging, depth=depth)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging, depth=depth)
        
        self.messages = [] # Initialize messages list
        self.query = None
        
        # Log depth info
        if enable_logging:
            print(f"[Depth {depth}] RLM_REPL initialized (max_depth={max_depth}, iterations={self._max_iterations})")
    
    def setup_context(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None):
        """
        Setup the context for the RLMClient.

        Args:
            context: The large context to analyze in the form of a list of messages, string, or Dict
            query: The user's question
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        # Initialize the conversation with the REPL prompt
        self.messages = build_system_prompt(depth=self.depth, max_depth=self.max_depth)
        self.logger.log_initial_messages(self.messages)
        
        # Initialize REPL environment with context data and depth info
        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data, 
            context_str=context_str, 
            recursive_model=self.recursive_model,
            depth=self.depth,
            max_depth=self.max_depth,
            max_iterations=self._max_iterations,  # Pass max_iterations to REPLEnv for sub-RLM creation
            api_key=self.api_key,
            enable_logging=self.logger.enabled,
        )
        
        return self.messages

    def completion(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None) -> str:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.
        """
        self.messages = self.setup_context(context, query)
        
        # Main loop runs for fixed # of root LM iterations
        for iteration in range(self._max_iterations):

            print(f"[Depth {self.depth}] Iteration {iteration}")
            
            # Query root LM to interact with REPL environment
            response = self.llm.completion(self.messages + [next_action_prompt(query, iteration)])

            print(f"[Depth {self.depth}] Response: {response}")
            
            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)
            
            # Process code execution or add assistant message
            if code_blocks is not None:
                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env, 
                    self.repl_env_logger, self.logger
                )
            else:
                # Add assistant message when there are no code blocks
                assistant_message = {"role": "assistant", "content": "You responded with:\n" + response}
                self.messages.append(assistant_message)
            
            # Check that model produced a final answer
            final_answer = utils.check_for_final_answer(
                response, self.repl_env, self.logger,
            )

            # In practice, you may need some guardrails here.
            if final_answer:
                self.logger.log_final_response(final_answer)
                return final_answer

            
        # If we reach here, no final answer was found in any iteration
        print(f"[Depth {self.depth}] No final answer found in any iteration")
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.logger.log_final_response(final_answer)

        return final_answer
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        raise NotImplementedError("Cost tracking not implemented for RLM REPL.")

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv(
            recursive_model=self.recursive_model,
            depth=self.depth,
            max_depth=self.max_depth,
            max_iterations=self._max_iterations,  # Pass max_iterations to REPLEnv
            api_key=self.api_key,
            enable_logging=self.logger.enabled,
        )
        self.messages = []
        self.query = None


if __name__ == "__main__":
    pass
