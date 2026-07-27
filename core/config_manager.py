import os
import json
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "system_config.json")

class CustomModel(BaseModel):
    id: str
    name: str
    provider: str = "openrouter"  # "openrouter" or "ollama"
    type: str = "llm"  # "llm" or "embedding"

def get_default_models() -> List[CustomModel]:
    return [
        CustomModel(id="google/gemini-2.0-flash-lite-001", name="⚡ Gemini 2.0 Flash Lite", provider="openrouter", type="llm"),
        CustomModel(id="meta-llama/llama-3.3-70b-instruct", name="🧠 Llama 3.3 70B Instruct", provider="openrouter", type="llm"),
        CustomModel(id="deepseek/deepseek-r1:free", name="🔬 DeepSeek R1 (Free)", provider="openrouter", type="llm"),
        CustomModel(id="anthropic/claude-3.5-sonnet", name="🎭 Claude 3.5 Sonnet", provider="openrouter", type="llm"),
        CustomModel(id="openai/gpt-4o-mini", name="⚡ GPT-4o Mini", provider="openrouter", type="llm"),
        CustomModel(id="llama3.2", name="🦙 Ollama Llama 3.2 (Local)", provider="ollama", type="llm"),
        CustomModel(id="mistral", name="🌋 Ollama Mistral (Local)", provider="ollama", type="llm"),
        CustomModel(id="nomic-embed-text", name="📐 Ollama Nomic Embed (Local)", provider="ollama", type="embedding"),
        CustomModel(id="nvidia/nemotron-3-embed-1b:free", name="📐 Nemotron Embed 1B (Free)", provider="openrouter", type="embedding"),
    ]

class SystemConfig(BaseModel):
    # API & Provider Config
    provider: str = Field(default="openrouter", description="openrouter or ollama")
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    
    # Selected Models
    llm_model: str = Field(default="google/gemini-2.0-flash-lite-001")
    embedding_model: str = Field(default="nvidia/nemotron-3-embed-1b:free")
    
    # Registered Models List
    custom_models: List[CustomModel] = Field(default_factory=get_default_models)

    # Retrieval Tuning Controls
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="1.0 = Dense only, 0.0 = BM25 only")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0, description="1.0 = Max relevance, 0.0 = Max diversity")
    top_k_candidates: int = Field(default=20, ge=5, le=100)
    top_n_final: int = Field(default=5, ge=1, le=20)
    
    # Feature Toggles
    enable_reranker: bool = Field(default=False)
    enable_hyde: bool = Field(default=False)
    strict_evidence: bool = Field(default=True)
    confirm_deletion: bool = Field(default=True)

class ConfigManager:
    """Central configuration manager for system settings."""

    def __init__(self):
        self.file_path = os.path.abspath(CONFIG_FILE_PATH)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.config = self._load_from_file()


    def _load_from_file(self) -> SystemConfig:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Refresh API key from env if empty in file
                    if not data.get("openrouter_api_key"):
                        data["openrouter_api_key"] = os.getenv("OPENROUTER_API_KEY", "")
                    return SystemConfig(**data)
            except Exception as e:
                logger.warning(f"Could not load config file ({e}). Initializing default config.")
        
        cfg = SystemConfig()
        self._save_to_file(cfg)
        return cfg

    def _save_to_file(self, cfg: SystemConfig = None) -> None:
        if cfg is None:
            cfg = self.config
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(cfg.model_dump(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config file ({self.file_path}): {e}")

    def update_config(self, updates: Dict[str, Any]) -> SystemConfig:
        current_dict = self.config.model_dump()
        current_dict.update({k: v for k, v in updates.items() if v is not None})
        self.config = SystemConfig(**current_dict)
        self._save_to_file()
        return self.config

    def add_model(self, model: CustomModel) -> SystemConfig:
        existing = [m for m in self.config.custom_models if m.id == model.id]
        if existing:
            existing[0].name = model.name
            existing[0].provider = model.provider
            existing[0].type = model.type
        else:
            self.config.custom_models.append(model)
        self._save_to_file()
        return self.config

    def remove_model(self, model_id: str) -> SystemConfig:
        self.config.custom_models = [m for m in self.config.custom_models if m.id != model_id]
        self._save_to_file()
        return self.config

    def reset_config(self, clear_credentials: bool = False, clear_custom_models: bool = False) -> SystemConfig:
        self.config = SystemConfig()
        if clear_credentials:
            self.config.openrouter_api_key = ""
        if clear_custom_models:
            self.config.custom_models = get_default_models()
        self._save_to_file()
        return self.config

    def get_config(self) -> SystemConfig:
        if not self.config.openrouter_api_key:
            self.config.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        return self.config

