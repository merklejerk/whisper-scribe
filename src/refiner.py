import ollama
import asyncio
import json  # Added json import
from typing import List, Dict, Any, Optional
import src.config as config

# Updated system prompt for JSON output
DEFAULT_SYSTEM_PROMPT = """You are an expert transcription editor. Your task is to correct any mistakes, misspellings, or transcription artifacts in the provided text. Use the provided chat log context to improve accuracy. Preserve the original meaning and speaker intent. Fix only actual errors, do not rephrase unnecessarily.
Output a JSON object with a single key 'refined_text' containing the corrected transcription string."""

class TranscriptRefiner:
    def __init__(self, model_name: Optional[str] = config.REFINER_OLLAMA_MODEL):
        self.model_name = model_name
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.enabled = bool(self.model_name)
        self._client: Optional[ollama.AsyncClient] = None

        if self.enabled:
            # Directly check model and create client, allowing exceptions to propagate
            ollama.show(self.model_name)  # Will raise if model not found or Ollama unavailable
            self._client = ollama.AsyncClient()
            print(f"Transcript Refiner enabled with Ollama model: {self.model_name}")
        else:
            print("Transcript Refiner disabled (no model configured).")


    async def refine(self, transcription: str, context_log: List[Dict[str, Any]]) -> str:
        """
        Refines a raw transcription using an Ollama model and recent log context.
        If the refiner is disabled, returns the original transcription.
        If an Ollama error occurs, the exception will propagate.

        Args:
            transcription: The raw transcription text to refine.
            context_log: A list of recent log entries (dictionaries) for context.

        Returns:
            The refined transcription string.
        """
        if not self.enabled or not self._client or not transcription:
            return transcription

        context_str = "\n".join([f"{entry['user_name']}: {entry['content']}" for entry in context_log])

        prompt = f"""Chat Log Context:
---
{context_str}
---
Transcription to Refine:
---
{transcription}
---"""

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
        ]

        # Call chat with format='json'
        response = await self._client.chat(
            model=self.model_name,
            messages=messages,
            format='json'  # Request JSON output
        )
        
        # Parse the JSON response
        response_content = response['message']['content']
        try:
            refined_data = json.loads(response_content)
            refined_text = refined_data.get('refined_text', '').strip()
            if not refined_text:
                 print(f"Warning: Ollama returned empty or invalid JSON structure: {response_content}")
                 return transcription  # Fallback if JSON key is missing or empty
        except json.JSONDecodeError:
            print(f"Error: Ollama response was not valid JSON: {response_content}")
            # Decide how to handle invalid JSON - raise error or return original?
            # For now, returning original to avoid breaking the flow, but logging error.
            return transcription 

        print(f"Original: '{transcription}' -> Refined: '{refined_text}'")
        return refined_text

