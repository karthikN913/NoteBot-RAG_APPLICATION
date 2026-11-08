"""Session-Isolated Chat History Management for NoteBot."""

from typing import List, Dict, Any, Optional

DEFAULT_WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "👋 Hi! I'm **NoteBot**, your AI study tutor and document intelligence assistant. "
               "Upload one or more PDF documents in the sidebar, and ask me any question or summary request. "
               "I remember our full conversational context across turns!",
    "sources": None
}


class ChatHistoryManager:
    """Manages chat messages for a user session."""

    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None):
        self.messages: List[Dict[str, Any]] = messages if messages is not None else [dict(DEFAULT_WELCOME_MESSAGE)]

    def add_user_message(self, content: str) -> None:
        """Appends a user message to the conversation."""
        self.messages.append({
            "role": "user",
            "content": content,
            "sources": None
        })

    def add_assistant_message(self, content: str, sources: Optional[List[Dict[str, Any]]] = None) -> None:
        """Appends an assistant response with optional source attribution."""
        self.messages.append({
            "role": "assistant",
            "content": content,
            "sources": sources
        })

    def pop_last_user_message(self) -> Optional[Dict[str, Any]]:
        """Removes the last user message if generation failed or aborted."""
        if self.messages and self.messages[-1].get("role") == "user":
            return self.messages.pop()
        return None

    def clear(self) -> None:
        """Resets the conversation to the initial welcome message."""
        self.messages = [dict(DEFAULT_WELCOME_MESSAGE)]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Returns the full message list."""
        return self.messages
