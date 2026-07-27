import httpx
import logging
from typing import List, Dict, Any, Optional
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ModelRouter:
    """Unified API client router for OpenRouter and local Ollama endpoints."""
    
    def __init__(self):
        self.config_mgr = ConfigManager.get_instance()

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 3072
    ) -> Dict[str, Any]:
        cfg = self.config_mgr.get_config()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n=== [LLM REQUEST] Provider: {cfg.provider} | Model: {cfg.llm_model} | Temp: {temperature} ===")
            if system_prompt:
                logger.debug(f"SYSTEM PROMPT:\n{system_prompt}")
            logger.debug(f"USER PROMPT & CONTEXT:\n{prompt}\n==================================================")

        if cfg.provider == "ollama":
            res = await self._call_ollama_chat(cfg, messages, temperature)
        else:
            res = await self._call_openrouter_chat(cfg, messages, temperature, max_tokens)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n=== [LLM RESPONSE] Model: {res.get('model')} ===")
            logger.debug(f"CONTENT:\n{res.get('content')}\n==================================================")

        return res

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        cfg = self.config_mgr.get_config()
        if cfg.provider == "ollama":
            return await self._call_ollama_embeddings(cfg, texts)
        else:
            return await self._call_openrouter_embeddings(cfg, texts)

    async def _call_openrouter_chat(
        self, cfg, messages: List[Dict[str, str]], temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        api_key = cfg.openrouter_api_key
        if not api_key:
            return {
                "content": "⚠️ **OPENROUTER_API_KEY is missing.** Please enter your OpenRouter API Key in the left sidebar Control Center, or switch provider to Local Ollama.",
                "model": cfg.llm_model,
                "provider": "openrouter",
                "usage": {"total_tokens": 0}
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/DarkWolfHunter007/Ultimate-RAG-System",
            "X-Title": "Ultimate RAG System by Allen MT Maliyil",
            "Content-Type": "application/json"
        }
        payload = {
            "model": cfg.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{cfg.openrouter_base_url}/chat/completions", headers=headers, json=payload)
                if resp.status_code != 200:
                    err_msg = f"HTTP {resp.status_code}"
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("error", {}).get("message", resp.text)
                    except Exception:
                        pass
                    return {
                        "content": f"⚠️ **OpenRouter API Error [{resp.status_code}]:** {err_msg}\n\n*Check your API key in Settings.*",
                        "model": cfg.llm_model,
                        "provider": "openrouter",
                        "usage": {"total_tokens": 0}
                    }
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "content": content,
                    "model": cfg.llm_model,
                    "provider": "openrouter",
                    "usage": usage
                }
        except httpx.TimeoutException:
            return {
                "content": f"⚠️ **Request Timeout:** OpenRouter API did not respond within 60 seconds. Please try again or switch model.",
                "model": cfg.llm_model,
                "provider": "openrouter",
                "usage": {"total_tokens": 0}
            }
        except Exception as e:
            return {
                "content": f"⚠️ **Connection Error:** {str(e)}",
                "model": cfg.llm_model,
                "provider": "openrouter",
                "usage": {"total_tokens": 0}
            }

    async def _call_openrouter_embeddings(self, cfg, texts: List[str]) -> List[List[float]]:
        api_key = cfg.openrouter_api_key
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Please configure your OpenRouter API key.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/DarkWolfHunter007/Ultimate-RAG-System",
            "X-Title": "Ultimate RAG System by Allen MT Maliyil",
            "Content-Type": "application/json"
        }
        payload = {
            "model": cfg.embedding_model,
            "input": texts
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{cfg.openrouter_base_url}/embeddings", headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter embedding error [{resp.status_code}]: {resp.text}")
            data = resp.json()
            embeddings = [item["embedding"] for item in data.get("data", [])]
            return embeddings

    async def _call_ollama_chat(self, cfg, messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
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

    async def _call_ollama_embeddings(self, cfg, texts: List[str]) -> List[List[float]]:
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
