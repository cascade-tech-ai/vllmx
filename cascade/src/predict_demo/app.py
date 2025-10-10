from __future__ import annotations

import asyncio
import codecs
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vllm.transformers_utils.tokenizer import (AnyTokenizer, decode_tokens,
                                               encode_tokens,
                                               get_cached_tokenizer,
                                               get_tokenizer as load_tokenizer)

_DEFAULT_TIMEOUT = 120.0


logger = logging.getLogger(__name__)


_TOKENIZER_CACHE: dict[str, AnyTokenizer] = {}
_MODEL_RESOLUTION_CACHE: dict[str, str] = {}
_TOKENIZER_LOCK = asyncio.Lock()

DEFAULT_SAMPLE_NAME = "snake_game.py"
SAMPLE_PROMPTS: dict[str, str] = {
    "snake_game.py": "make it multiplayer",
    "client_database.html": "add a filter dropdown to choose client status",
}

SYSTEM_PROMPT = (
    "Each user message will be a user request to modify some code or other type "
    "of text document. Output ONLY the modified code, no other text or "
    "additional markdown formatting. The input document is below in <input_code> "
    "tags, but you should output the code without those tags, just the text "
    "itself."
)


class ChatStreamPayload(BaseModel):
    document: str = ""
    prompt: str = ""
    temperature: float = 0.0
    use_prediction: bool = True
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    model: Optional[str] = None


class TokenizePayload(BaseModel):
    text: str
    add_special_tokens: bool = False
    model: Optional[str] = None


class DetokenizePayload(BaseModel):
    tokens: list[int]
    add_special_tokens: bool = False
    model: Optional[str] = None


def _render_input_code(document: str) -> Optional[str]:
    if not document:
        return None
    cleaned = document.strip("\n")
    if not cleaned:
        return None
    return f"<input_code>\n{cleaned}\n</input_code>"


def _load_messages(document: str, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    parts: list[str] = []

    rendered = _render_input_code(document)
    if rendered:
        parts.append(rendered)

    stripped_prompt = prompt.strip()
    if stripped_prompt:
        parts.append(stripped_prompt)

    if parts:
        messages.append({"role": "user", "content": "\n\n".join(parts)})

    return messages


def _resolve_samples_dir() -> Path:
    return Path(__file__).resolve().parent / "samples"


@lru_cache(maxsize=1)
def _samples_dir() -> Path:
    return _resolve_samples_dir()


def _safe_sample_path(sample_name: str) -> Path:
    base = _samples_dir()
    target = (base / sample_name).resolve()
    try:
        target.relative_to(base)
    except ValueError:  # pragma: no cover - guards path traversal
        raise FileNotFoundError(sample_name)
    if not target.is_file():
        raise FileNotFoundError(sample_name)
    return target


def _sample_metadata() -> tuple[Optional[str], dict[str, str]]:
    samples_dir = _samples_dir()
    available: set[str] = set()
    if samples_dir.exists():
        for path in samples_dir.iterdir():
            if path.is_file():
                available.add(path.name)
    prompts = {
        name: text
        for name, text in SAMPLE_PROMPTS.items()
        if name in available
    }
    default: Optional[str]
    if DEFAULT_SAMPLE_NAME in available:
        default = DEFAULT_SAMPLE_NAME
    elif available:
        default = sorted(available)[0]
    else:
        default = None
    return default, prompts


def _utf16_length(text: str) -> int:
    if not text:
        return 0
    # UTF-16 code unit length matches JavaScript string length semantics.
    return len(text.encode("utf-16-le")) // 2


async def _resolve_model_uri(server_url: str, timeout: httpx.Timeout,
                             model_alias: str) -> str:
    cached = _MODEL_RESOLUTION_CACHE.get(model_alias)
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(f"{server_url}/v1/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch models list: {exc}") from exc

    payload = response.json()
    items = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        raise RuntimeError("Malformed /v1/models response")

    for entry in items:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        if str(entry_id) != model_alias:
            continue
        model_uri = (entry.get("root") or entry.get("model")
                     or entry_id)
        if not model_uri:
            break
        resolved = str(model_uri)
        _MODEL_RESOLUTION_CACHE[model_alias] = resolved
        return resolved

    raise RuntimeError(f"Model alias '{model_alias}' not found in /v1/models")


async def _get_tokenizer(server_url: str, timeout: httpx.Timeout,
                         model_alias: str) -> AnyTokenizer:
    cached = _TOKENIZER_CACHE.get(model_alias)
    if cached is not None:
        return cached

    async with _TOKENIZER_LOCK:
        cached = _TOKENIZER_CACHE.get(model_alias)
        if cached is not None:
            return cached

        logger.info("Fetching tokenizer model for alias '%s'", model_alias)
        model_uri = await _resolve_model_uri(server_url, timeout,
                                             model_alias)
        logger.info("Loading tokenizer for model '%s'", model_uri)
        try:
            tokenizer = await asyncio.to_thread(load_tokenizer,
                                                model_uri,
                                                trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load tokenizer: {exc}") from exc

        cached_tokenizer = get_cached_tokenizer(tokenizer)
        _TOKENIZER_CACHE[model_alias] = cached_tokenizer
        logger.info("Tokenizer loaded for model '%s'", model_uri)
        return cached_tokenizer


class DebugStaticFiles(StaticFiles):

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


def create_app(server_url: str, default_model: str, *, debug: bool = False) -> FastAPI:
    app = FastAPI(title="Predict Demo", version="0.1.0")
    app.state.server_url = server_url.rstrip("/")
    app.state.default_model = default_model
    app.state.http_timeout = httpx.Timeout(_DEFAULT_TIMEOUT)

    @app.on_event("startup")
    async def _warm_tokenizer() -> None:
        try:
            await _get_tokenizer(app.state.server_url, app.state.http_timeout,
                                 app.state.default_model)
        except Exception as exc:  # pragma: no cover - startup best-effort
            logger.info("Tokenizer warmup skipped: %s", exc)

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        default_sample, prompts = _sample_metadata()
        return JSONResponse({
            "server_url": app.state.server_url,
            "model": app.state.default_model,
            "default_sample": default_sample,
            "sample_prompts": prompts,
        })

    @app.get("/api/samples")
    async def list_samples() -> JSONResponse:
        samples_dir = _samples_dir()
        default_sample, prompts = _sample_metadata()
        if not samples_dir.exists():
            return JSONResponse({
                "samples": [],
                "default": default_sample,
                "sample_prompts": prompts,
            })
        items = []
        for path in sorted(samples_dir.iterdir()):
            if not path.is_file():
                continue
            rel_name = path.name
            display = path.stem.replace("_", " ").title()
            items.append({"id": rel_name, "label": display})
        return JSONResponse({
            "samples": items,
            "default": default_sample,
            "sample_prompts": prompts,
        })

    @app.get("/api/samples/{sample_name}")
    async def get_sample(sample_name: str) -> JSONResponse:
        try:
            sample_path = _safe_sample_path(sample_name)
        except FileNotFoundError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404,
                                detail=f"Sample '{sample_name}' not found")
        try:
            content = sample_path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - unlikely IO error
            raise HTTPException(status_code=500,
                                detail=f"Failed to read sample: {exc}")
        return JSONResponse({
            "id": sample_name,
            "label": sample_path.stem.replace("_", " ").title(),
            "content": content,
        })

    @app.post("/api/tokenize")
    async def tokenize(payload: TokenizePayload) -> JSONResponse:
        if not payload.text:
            return JSONResponse({"tokens": [], "token_strs": []})
        request_model = payload.model or app.state.default_model
        try:
            tokenizer = await _get_tokenizer(app.state.server_url,
                                             app.state.http_timeout,
                                             request_model)
        except Exception as exc:
            raise HTTPException(status_code=502,
                                detail=f"Tokenizer unavailable: {exc}") from exc

        try:
            token_ids = await asyncio.to_thread(
                encode_tokens,
                tokenizer,
                payload.text,
                add_special_tokens=payload.add_special_tokens,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500,
                                detail=f"Encoding failed: {exc}") from exc

        token_strs = tokenizer.convert_ids_to_tokens(token_ids)
        return JSONResponse({"tokens": token_ids, "token_strs": token_strs})

    async def _augment_sse_event(raw_event: str,
                                 tokenizer: Optional[AnyTokenizer]) -> str:
        if not raw_event or tokenizer is None:
            return raw_event

        lines = raw_event.split("\n")
        data_lines = []
        passthrough_lines = []
        for line in lines:
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            else:
                passthrough_lines.append(line)

        if not data_lines:
            return raw_event

        data_payload = "\n".join(data_lines)
        if data_payload.strip() == "[DONE]":
            return raw_event

        try:
            payload_obj = json.loads(data_payload)
        except json.JSONDecodeError:
            return raw_event

        choices = payload_obj.get("choices")
        if not isinstance(choices, list):
            return raw_event

        modified = False

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            delta_content = delta.get("content")
            segments: list[str] = []
            if isinstance(delta_content, str):
                segments.append(delta_content)
            elif isinstance(delta_content, list):
                for part in delta_content:
                    if (isinstance(part, dict) and "text" in part and
                            part["text"] is not None):
                        segments.append(str(part["text"]))

            chunk_text = "".join(segments)
            proposed_raw = choice.get("spec_tokens_proposed")
            accepted_raw = choice.get("spec_tokens_accepted")
            try:
                proposed = int(proposed_raw)
            except (TypeError, ValueError):
                proposed = 0
            try:
                accepted_tokens = int(accepted_raw)
            except (TypeError, ValueError):
                accepted_tokens = 0

            if proposed <= 0 and accepted_tokens <= 0:
                continue

            if not chunk_text:
                choice["spec_chars_accepted"] = 0
                modified = True
                continue

            if accepted_tokens <= 0:
                choice["spec_chars_accepted"] = 0
                modified = True
                continue

            try:
                token_ids = await asyncio.to_thread(
                    encode_tokens,
                    tokenizer,
                    chunk_text,
                    add_special_tokens=False,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Tokenization failed for chunk: %s", exc)
                choice["spec_chars_accepted"] = 0
                modified = True
                continue

            if not token_ids:
                choice["spec_chars_accepted"] = 0
                modified = True
                continue

            accepted_count = max(0, min(accepted_tokens, len(token_ids)))
            if accepted_count == 0:
                choice["spec_chars_accepted"] = 0
                modified = True
                continue

            accepted_ids = token_ids[:accepted_count]
            try:
                accepted_text = await asyncio.to_thread(
                    decode_tokens,
                    tokenizer,
                    accepted_ids,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Detokenization failed for chunk: %s", exc)
                choice["spec_chars_accepted"] = 0
                modified = True
                continue

            if not chunk_text.startswith(accepted_text):
                choice["spec_chars_accepted"] = 0
                modified = True
                continue

            accepted_chars = _utf16_length(accepted_text)
            total_chars = _utf16_length(chunk_text)
            choice["spec_chars_accepted"] = min(accepted_chars, total_chars)
            modified = True

        if not modified:
            return raw_event

        serialized = json.dumps(payload_obj, separators=(",", ":"))
        if passthrough_lines:
            passthrough = [line for line in passthrough_lines if line]
            passthrough.append(f"data: {serialized}")
            return "\n".join(passthrough)
        return f"data: {serialized}"

    @app.post("/api/chat-stream")
    async def chat_stream(payload: ChatStreamPayload) -> StreamingResponse:
        messages = _load_messages(payload.document, payload.prompt)
        if len(messages) <= 1:
            raise HTTPException(status_code=400,
                                detail="Prompt or document required")
        model_name = payload.model or app.state.default_model
        request_json: dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "temperature": payload.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if payload.use_prediction:
            request_json["prediction"] = {
                "type": "content",
                "content": payload.document,
            }
        if payload.max_completion_tokens is not None:
            request_json["max_completion_tokens"] = payload.max_completion_tokens

        try:
            tokenizer = await _get_tokenizer(app.state.server_url,
                                             app.state.http_timeout,
                                             model_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tokenizer unavailable for streaming: %s", exc)
            tokenizer = None

        async def event_iterator() -> AsyncIterator[bytes]:
            decoder = codecs.getincrementaldecoder("utf-8")()
            buffer = ""

            async with httpx.AsyncClient(timeout=None) as client:
                try:
                    async with client.stream(
                            "POST",
                            f"{app.state.server_url}/v1/chat/completions",
                            json=request_json) as upstream:
                        upstream.raise_for_status()

                        async for chunk in upstream.aiter_raw():
                            if not chunk:
                                continue
                            text = decoder.decode(chunk, final=False)
                            if not text:
                                continue
                            buffer += text.replace("\r\n", "\n")

                            while True:
                                sep_index = buffer.find("\n\n")
                                if sep_index == -1:
                                    break
                                raw_event = buffer[:sep_index]
                                buffer = buffer[sep_index + 2:]
                                processed = await _augment_sse_event(
                                    raw_event, tokenizer)
                                yield (processed + "\n\n").encode("utf-8")

                        remainder = decoder.decode(b"", final=True)
                        if remainder:
                            buffer += remainder.replace("\r\n", "\n")

                        while buffer:
                            sep_index = buffer.find("\n\n")
                            if sep_index == -1:
                                processed = await _augment_sse_event(
                                    buffer, tokenizer)
                                yield (processed + "\n\n").encode("utf-8")
                                break
                            raw_event = buffer[:sep_index]
                            buffer = buffer[sep_index + 2:]
                            processed = await _augment_sse_event(
                                raw_event, tokenizer)
                            yield (processed + "\n\n").encode("utf-8")

                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text or exc.response.reason_phrase
                    message = f"Upstream error {exc.response.status_code}: {detail}"
                    raise HTTPException(status_code=exc.response.status_code,
                                        detail=message)
                except httpx.HTTPError as exc:
                    raise HTTPException(status_code=502,
                                        detail=f"Streaming failed: {exc}")

        return StreamingResponse(event_iterator(),
                                 media_type="text/event-stream")

    @app.post("/api/detokenize")
    async def detokenize(payload: DetokenizePayload) -> JSONResponse:
        if not payload.tokens:
            return JSONResponse({"text": ""})
        request_model = payload.model or app.state.default_model
        try:
            tokenizer = await _get_tokenizer(app.state.server_url,
                                             app.state.http_timeout,
                                             request_model)
        except Exception as exc:
            raise HTTPException(status_code=502,
                                detail=f"Tokenizer unavailable: {exc}") from exc

        try:
            token_ids = [int(tok) for tok in payload.tokens]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400,
                                detail=f"Invalid token ids: {exc}") from exc

        try:
            text = await asyncio.to_thread(
                decode_tokens,
                tokenizer,
                token_ids,
                skip_special_tokens=not payload.add_special_tokens,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500,
                                detail=f"Decoding failed: {exc}") from exc

        return JSONResponse({"text": text})

    static_dir = Path(__file__).resolve().parent / "static"
    static_class = DebugStaticFiles if debug else StaticFiles
    app.mount("/", static_class(directory=static_dir, html=True), name="static")

    return app
