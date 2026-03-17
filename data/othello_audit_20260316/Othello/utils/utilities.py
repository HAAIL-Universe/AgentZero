import asyncio
import logging

async def async_retry(coro_func, *args, retries=3, delay=1, **kwargs):
    """
    Awaits an async function with automatic retries.
    
    Usage:
        await async_retry(self.model.chat, messages)

    Args:
        coro_func: The coroutine function to call.
        *args, **kwargs: Arguments for the coroutine.
        retries: Number of retry attempts.
        delay: Delay between retries (seconds).
    """
    for attempt in range(1, retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Retry {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
