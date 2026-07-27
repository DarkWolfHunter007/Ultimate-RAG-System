import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

CHAT_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chat_sessions.json")

class ChatManager:

    def __init__(self):
        self.file_path = os.path.abspath(CHAT_FILE_PATH)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.chats: Dict[str, Dict[str, Any]] = self._load_chats()


    def _load_chats(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load chat sessions ({e}). Initializing empty store.")
        return {}

    def _save_chats(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.chats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat sessions ({self.file_path}): {e}")

    def list_chats(self) -> List[Dict[str, Any]]:
        summaries = []
        for chat_id, data in self.chats.items():
            messages = data.get("messages", [])
            last_time = messages[-1].get("timestamp", data.get("created_at")) if messages else data.get("created_at")
            summaries.append({
                "id": chat_id,
                "title": data.get("title", "New Chat"),
                "created_at": data.get("created_at", 0),
                "updated_at": last_time,
                "message_count": len(messages)
            })
        # Sort newest updated first
        summaries.sort(key=lambda x: x["updated_at"], reverse=True)
        return summaries

    def create_chat(self, title: Optional[str] = None) -> Dict[str, Any]:
        chat_id = f"chat_{int(time.time() * 1000)}"
        if not title:
            count = len(self.chats) + 1
            title = f"Chat Session #{count}"
        
        chat_data = {
            "id": chat_id,
            "title": title,
            "created_at": time.time(),
            "messages": []
        }
        self.chats[chat_id] = chat_data
        self._save_chats()
        return chat_data

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        return self.chats.get(chat_id)

    def delete_chat(self, chat_id: str) -> bool:
        if chat_id in self.chats:
            del self.chats[chat_id]
            self._save_chats()
            return True
        return False

    def add_message(self, chat_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        if chat_id not in self.chats:
            self.create_chat()

        chat = self.chats[chat_id]
        if "messages" not in chat:
            chat["messages"] = []

        message["timestamp"] = time.time()
        chat["messages"].append(message)

        # Auto-update title from first user query if generic title
        if len(chat["messages"]) == 1 and message.get("role") == "user":
            user_text = message.get("content", "").strip()
            if user_text:
                short_title = user_text[:35] + ("..." if len(user_text) > 35 else "")
                chat["title"] = short_title

        self._save_chats()
        return chat

    def clear_all(self) -> None:
        self.chats = {}
        self._save_chats()
