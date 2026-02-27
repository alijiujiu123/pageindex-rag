"""LLM wrapper supporting OpenAI-compatible APIs (OpenAI, OpenRouter, etc.)."""

import asyncio
import logging
import time

import openai

logger = logging.getLogger(__name__)

_MAX_RETRIES = 10


def llm_call(model: str, prompt: str, api_key: str, base_url: str) -> str:
    """Synchronous LLM call with retry."""
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    for i in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call error (attempt {i+1}): {e}")
            if i < _MAX_RETRIES - 1:
                time.sleep(1)
    return "Error"


async def llm_call_async(model: str, prompt: str, api_key: str, base_url: str) -> str:
    """Async LLM call with retry."""
    for i in range(_MAX_RETRIES):
        try:
            async with openai.AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Async LLM call error (attempt {i+1}): {e}")
            if i < _MAX_RETRIES - 1:
                await asyncio.sleep(1)
    return "Error"
