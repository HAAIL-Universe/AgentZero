import os
import openai
import logging
from typing import Optional
from dotenv import load_dotenv

def pick_model(default: str = "gpt-4o"):
    print("Which model would you like FELLO to use this session?")
    print("1: gpt-3.5-turbo")
    print("2: gpt-4")
    print("3: gpt-4-turbo")
    print("4: gpt-4o")
    choice = input(f"Enter 1-4 [default={default}]: ").strip()
    return {
        "1": "gpt-3.5-turbo",
        "2": "gpt-4",
        "3": "gpt-4-turbo",
        "4": "gpt-4o"
    }.get(choice, default)


class LLMWrapper:
    def __init__(self, model: Optional[str] = None):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.client = openai.OpenAI(api_key=api_key)

        # Check for cached model
        model_cache_path = ".llm_model_cache"
        if model:
            self.model = model
            with open(model_cache_path, "w") as f:
                f.write(model)
        elif os.path.exists(model_cache_path):
            with open(model_cache_path, "r") as f:
                self.model = f.read().strip()
        else:
            self.model = pick_model()
            with open(model_cache_path, "w") as f:
                f.write(self.model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 300) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logging.debug("Making API call to OpenAI...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=10  # Set timeout for 10 seconds
            )
            logging.debug("API call completed.")
            response_content = response.choices[0].message.content
            if response_content is not None:
                return response_content.strip()
            else:
                return "[LLM Error]: No content returned from LLM."
        except Exception as e:
            logging.error(f"Error during LLM API call: {str(e)}")
            return f"[LLM Error]: {str(e)}"

class AsyncLLMWrapper(LLMWrapper):
    async def chat(self, messages):
        # messages is a list of dicts: [{role: ..., content: ...}]
        # Pull out system and user prompt for .generate()
        system_prompt = None
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            elif msg.get("role") == "user":
                user_prompt = msg.get("content")
        # Run synchronous generate() in a thread so async works
        import asyncio
        return await asyncio.to_thread(self.generate, user_prompt, system_prompt)
