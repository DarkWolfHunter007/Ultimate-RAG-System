import httpx
import json
import logging
from typing import Any, Optional, AsyncGenerator
from core.config_manager import ConfigManager
from analytics.telemetry_logger import track

logger = logging.getLogger(__name__)


class ModelRouter:
    """Unified API client router for OpenRouter and local Ollama endpoints with streaming support."""

    def __init__(self):
        self.config_mgr = ConfigManager()

    @track(name="llm_generate_completion")
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 3072
    ) -> dict[str, Any]:
        cfg = self.config_mgr.get_config()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if cfg.provider == "ollama":
            res = await self._call_ollama_chat(cfg, messages, temperature)
        else:
            res = await self._call_openrouter_chat(cfg, messages, temperature, max_tokens)

        return res

    async def stream_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 3072
    ) -> AsyncGenerator[str, None]:
        """Streams completion tokens word-by-word (OpenAI SSE or Ollama JSON stream)."""
        cfg = self.config_mgr.get_config()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if cfg.provider == "ollama":
            async for chunk in self._call_ollama_stream(cfg, messages, temperature):
                yield chunk
        else:
            async for chunk in self._call_openrouter_stream(cfg, messages, temperature, max_tokens):
                yield chunk

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        cfg = self.config_mgr.get_config()
        if cfg.provider == "ollama":
            return await self._call_ollama_embeddings(cfg, texts)
        else:
            return await self._call_openrouter_embeddings(cfg, texts)

    def _openrouter_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/DarkWolfHunter007/Ultimate-RAG-System",
            "X-Title": "Ultimate RAG System by Allen MT Maliyil",
            "Content-Type": "application/json"
        }

    async def _call_openrouter_chat(
        self, cfg, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        api_key = cfg.openrouter_api_key
        def _err(msg: str) -> dict[str, Any]:
            return {"content": msg, "model": cfg.llm_model, "provider": "openrouter", "usage": {"total_tokens": 0}}

        if not api_key:
            return _err("⚠️ **OPENROUTER_API_KEY is missing.** Please enter your OpenRouter API Key in the left sidebar Control Center, or switch provider to Local Ollama.")

        payload = {
            "model": cfg.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{cfg.openrouter_base_url}/chat/completions",
                    headers=self._openrouter_headers(api_key),
                    json=payload
                )
                if resp.status_code != 200:
                    try:
                        err_msg = resp.json().get("error", {}).get("message", resp.text)
                    except Exception:
                        err_msg = f"HTTP {resp.status_code}"
                    return _err(f"⚠️ **OpenRouter API Error [{resp.status_code}]:** {err_msg}")
                data = resp.json()
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "model": cfg.llm_model,
                    "provider": "openrouter",
                    "usage": data.get("usage", {})
                }
        except httpx.TimeoutException:
            return _err("⚠️ **Request Timeout:** OpenRouter API did not respond within 60 seconds.")
        except Exception as e:
            return _err(f"⚠️ **Connection Error:** {str(e)}")

    async def _call_openrouter_stream(
        self, cfg, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        api_key = cfg.openrouter_api_key
        if not api_key:
            yield "⚠️ **OPENROUTER_API_KEY is missing.** Please enter your API Key in Settings."
            return

        payload = {
            "model": cfg.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{cfg.openrouter_base_url}/chat/completions",
                    headers=self._openrouter_headers(api_key),
                    json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            raw_data = line[6:].strip()
                            if raw_data == "[DONE]":
                                break
                            try:
                                json_obj = json.loads(raw_data)
                                delta = json_obj["choices"][0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                pass
        except Exception as e:
            yield f"\n⚠️ **Streaming Error:** {str(e)}"

    async def _call_openrouter_embeddings(self, cfg, texts: list[str]) -> list[list[float]]:
        api_key = cfg.openrouter_api_key
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Please configure your OpenRouter API key.")

        payload = {"model": cfg.embedding_model, "input": texts}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{cfg.openrouter_base_url}/embeddings",
                headers=self._openrouter_headers(api_key),
                json=payload
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter embedding error [{resp.status_code}]: {resp.text}")
            return [item["embedding"] for item in resp.json().get("data", [])]

    async def _call_ollama_chat(self, cfg, messages: list[dict[str, str]], temperature: float) -> dict[str, Any]:
        url = f"{cfg.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": cfg.llm_model.split("/")[-1],
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API error [{resp.status_code}]: {resp.text}")
            data = resp.json()
            return {
                "content": data.get("message", {}).get("content", ""),
                "model": cfg.llm_model,
                "provider": "ollama",
                "usage": {"total_tokens": data.get("eval_count", 0)}
            }

    async def _call_ollama_stream(self, cfg, messages: list[dict[str, str]], temperature: float) -> AsyncGenerator[str, None]:
        url = f"{cfg.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": cfg.llm_model.split("/")[-1],
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": True
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                delta = data.get("message", {}).get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                pass
        except Exception as e:
            yield f"\n⚠️ **Ollama Stream Error:** {str(e)}"

    async def _call_ollama_embeddings(self, cfg, texts: list[str]) -> list[list[float]]:
        url = f"{cfg.ollama_base_url.rstrip('/')}/api/embeddings"
        model_name = cfg.embedding_model.split("/")[-1]
        embeddings = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                resp = await client.post(url, json={"model": model_name, "prompt": text})
                if resp.status_code == 200:
                    embeddings.append(resp.json().get("embedding", []))
                else:
                    raise RuntimeError(f"Ollama embedding error [{resp.status_code}] for model '{model_name}': {resp.text}")
        return embeddings
