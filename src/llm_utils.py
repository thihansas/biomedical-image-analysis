
import json
import re

import ollama

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# default num_ctx (131072 for llama3.2-vision) blows up the KV-cache estimate
# and can make the model refuse to load on ~16GB RAM machines; our prompts
# never get close to 4096 tokens anyway
DEFAULT_OPTIONS = {"num_ctx": 4096}


def _merge_options(options: dict | None) -> dict:
    return {**DEFAULT_OPTIONS, **(options or {})}


def extract_json(text: str) -> dict:
    """Pull the first {...} block out of a model response and parse it.

    Local models keep wrapping JSON in prose or markdown fences even when told
    not to, so we grab the brace block instead of trusting the whole response.
    """
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(match.group(0))


def chat_json(model: str, prompt: str, images: list[str] | None = None,
               max_retries: int = 3, options: dict | None = None) -> tuple[dict, str]:
    """Call an Ollama model, return (parsed_json, raw_text).

    Retries up to max_retries times on a JSON parse failure, then raises.
    """
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images

    last_error = None
    for attempt in range(max_retries):
        response = ollama.chat(model=model, messages=[message], options=_merge_options(options))
        raw_text = response["message"]["content"]
        try:
            parsed = extract_json(raw_text)
            return parsed, raw_text
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(
        f"Model '{model}' failed to return valid JSON after {max_retries} attempts"
    ) from last_error


def chat_text(model: str, prompt: str, images: list[str] | None = None,
               options: dict | None = None) -> str:
    """Call an Ollama model for a plain free-text response (e.g. a narrative)."""
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images

    response = ollama.chat(
        model=model,
        messages=[message],
        options=_merge_options(options),
    )
    return response["message"]["content"].strip()
