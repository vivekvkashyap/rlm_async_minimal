"""
OpenAI Client wrapper specifically for GPT models.
"""

import os
import asyncio
from typing import Optional, List
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")


class AsyncOpenAIClient:
    """Async OpenAI Client for parallel batch completions."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Async completion for a single request."""
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
    
    async def batch_completion(
        self,
        messages_list: List[list[dict[str, str]] | str],
        max_tokens: Optional[int] = None,
        max_concurrent: int = 10,
        **kwargs
    ) -> List[str]:
        """
        Batch completion for multiple requests in parallel.
        
        Args:
            messages_list: List of message inputs (each can be str or list of dicts)
            max_tokens: Maximum tokens per completion
            max_concurrent: Maximum number of concurrent requests (for rate limiting)
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            List of completion strings in the same order as input
        """
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_completion(messages):
            async with semaphore:
                return await self.completion(messages, max_tokens, **kwargs)
        
        # Run all completions concurrently with rate limiting
        tasks = [limited_completion(messages) for messages in messages_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error strings
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Error in batch request {i}: {str(result)}"
                processed_results.append(error_msg)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def batch_completion_sync(
        self,
        messages_list: List[list[dict[str, str]] | str],
        max_tokens: Optional[int] = None,
        max_concurrent: int = 10,
        **kwargs
    ) -> List[str]:
        """
        Synchronous wrapper for batch completion.
        
        This allows calling batch completions from synchronous code
        by internally managing the async event loop.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.batch_completion(messages_list, max_tokens, max_concurrent, **kwargs)
                    )
                    return future.result()
            else:
                # If no loop is running, use asyncio.run
                return asyncio.run(
                    self.batch_completion(messages_list, max_tokens, max_concurrent, **kwargs)
                )
        except RuntimeError:
            # If there's no event loop, create one
            return asyncio.run(
                self.batch_completion(messages_list, max_tokens, max_concurrent, **kwargs)
            )