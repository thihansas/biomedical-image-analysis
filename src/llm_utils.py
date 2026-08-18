
import json
import re

import ollama

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Ollama's models default to a very large context window (e.g. 131072 tokens for
# llama3.2-vision), which inflates the KV-cache memory estimate far beyond what
# these short prompts need and can make the model refuse to load on machines
# with ~16GB RAM. None of our prompts+images come close to 4096 tokens, so we
# cap it here for every call rather than relying on each caller to remember to.
DEFAULT_OPTIONS = {"num_ctx": 4096}


def _merge_options(options: dict | None) -> dict:
    return {**DEFAULT_OPTIONS, **(options or {})}


def extract_json(text: str) -> dict:
    """Pull the first {...} block out of a model response and parse it.

    Local LLMs frequently wrap JSON in prose or markdown fences even when
    asked not to, so we search for the outermost brace block rather than
    assuming the whole response is valid JSON.
    """
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(match.group(0))


def chat_json(model: str, prompt: str, images: list[str] | None = None,
               max_retries: int = 3, options: dict | None = None) -> tuple[dict, str]:
    """Call an Ollama model and return (parsed_json, raw_text).

    Retries up to `max_retries` times if the response cannot be parsed as
    JSON. Raises the last parsing error if all attempts fail.
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
    """Call an Ollama model for a free-text (non-JSON) response, e.g. a narrative."""
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images

    response = ollama.chat(
        model=model,
        messages=[message],
        options=_merge_options(options),
    )
    return response["message"]["content"].strip()
