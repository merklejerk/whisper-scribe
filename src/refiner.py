import ollama
import re
from pydantic import BaseModel, Field, ValidationError

from .config import (
    USERNAME_MAP,
    REFINER_OLLAMA_MODEL,
    REFINER_TEMPERATURE,
    REFINER_SYSTEM_PROMPT,
    REFINER_CONTEXT_LOG_LINES,
    REFINER_TIMEOUT,
)
from .logging import LogEntry

class RefinerResponse(BaseModel):
    edit_explanation: str = Field(description="Brief explanation of the edits you made.")
    corrected_contet: str = Field(description="What the speaker said (corrected).", default="")

class TranscriptRefiner:
    def __init__(self, model_name: str | None = REFINER_OLLAMA_MODEL):
        self.model_name = model_name
        self.system_prompt = REFINER_SYSTEM_PROMPT
        self._client: ollama.AsyncClient | None = None

        if self.model_name:
            ollama.show(self.model_name)  # Will raise if model not found or Ollama unavailable
            self._client = ollama.AsyncClient(timeout=REFINER_TIMEOUT)
            print(f"Transcript Refiner enabled with Ollama model: {self.model_name}")
        else:
            print("Transcript Refiner disabled (no model configured).")

    async def refine(
        self,
        transcription: LogEntry,
        context_log: list[LogEntry],
    ) -> str | None:
        """
        Refines a raw transcription using an Ollama model and recent log context.

        Args:
            transcription: The raw transcription to refine.
            context_log: A list of recent log entries for context.
            aliases: A dictionary of user name aliases.

        Returns:
            The refined transcription string.
        """
        if not self._client:
            return transcription

        print(f"Refining transcription from {transcription.user_name}...")
        
        aliases = {e.user_id: USERNAME_MAP.get(e.user_name, e.user_name) for e in context_log}
        prompt = f"""Past dialogue:
{"\n".join([f"> {aliases[e.user_id]}: {e.content}" for e in context_log]) if context_log else "No context available."}

Line to correct:
> {USERNAME_MAP.get(transcription.user_name, transcription.user_name)}: {transcription.content}
"""
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
        ]

        try:
            response = await self._client.chat(
                model=self.model_name,
                messages=messages,
                format=RefinerResponse.model_json_schema(),
                options={
                    "num_ctx": REFINER_CONTEXT_LOG_LINES * 500 + 2048,
                    "max_tokens": len(transcription.content) * 2 + 2048,
                    "temperature": REFINER_TEMPERATURE,
                },
            )
            raw_json = clean_response_json(response.message.content)
            response = RefinerResponse.model_validate_json(raw_json)
            refined = response.corrected_contet
        except ValidationError as e:
            print(f"Refiner response validation error: {e}")
            print(raw_json)
            return transcription.content
        except Exception as e:
            print(f"Error during refinement: {e} ({type(e)})")
            return transcription.content

        if refined != transcription.content and getattr(response, 'edit_explanation'):
            print(f"Refiner notes: {response.edit_explanation}")
        
        if not refined:
            return None
        
        # Remove leading 'NAME: ' or 'NAME - ' prefix if present
        refined = re.sub(r"^\s*[^:]+\s*[:\-]\s*", "", refined, count=1)
        if not re.search(r"[a-z0-9]+", refined):
            return None
        # Replace ellipsis character with standard three dots.
        refined = refined.replace("…", "...")

        # If the refined text is identical to the last log entry, return None.
        if context_log and context_log[-1].content == refined:
            print("Refiner output matches last log entry, skipping.")
            return None
        
        return refined


def clean_response_json(raw_json: str) -> str:
    """Remove common artifacts from JSON responses."""
    # Remove anything before the first '{' and after the last '}'
    cleaned_json = re.sub(r"^.*?({.*?})\s*.*$", r"\1", raw_json, flags=re.DOTALL)
    # Remove weird quote characters.
    cleaned_json = re.sub(r"[“”]", '"', cleaned_json)
    return cleaned_json