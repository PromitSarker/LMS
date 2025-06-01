import os
import logging
import httpx
from dotenv import load_dotenv
from App.core.config import settings
from typing import List, Dict
from collections import defaultdict

load_dotenv()

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.base_url = settings.GROQ_API_URL
        self.model = settings.LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.default_language = settings.DEFAULT_LANGUAGE
        # Dictionary to store conversation history for each user
        self.conversations = defaultdict(list)
        # Maximum number of messages to keep in history
        self.max_history = 10

    async def get_chat_response(self, user_message: str, user_id: str = "default", language: str = None) -> str:
        try:
            # Use specified language or default
            lang = language if language in settings.SUPPORTED_LANGUAGES else self.default_language
            
            # Get conversation history for this user
            conversation = self.conversations[user_id]
            
            # Add system prompt and conversation history
            messages = [
                {
                    "role": "system",
                    "content": settings.SYSTEM_PROMPTS[lang]
                }
            ]
            
            # Add conversation history
            messages.extend(conversation)
            
            # Add current user message
            current_message = {
                "role": "user",
                "content": f"Respond in {lang}: {user_message}"
            }
            messages.append(current_message)
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
                
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.text}")
                return f"Sorry, I encountered an error: {response.text}"
                
            data = response.json()
            assistant_message = data["choices"][0]["message"]["content"]
            
            # Store the conversation
            self.conversations[user_id].append(current_message)
            self.conversations[user_id].append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Keep only the last N messages
            if len(self.conversations[user_id]) > self.max_history:
                self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"ChatService error: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def clear_history(self, user_id: str = "default") -> None:
        """Clear conversation history for a specific user"""
        if user_id in self.conversations:
            self.conversations[user_id].clear()
            logger.info(f"Cleared conversation history for user {user_id}")

    def get_history(self, user_id: str = "default") -> List[Dict[str, str]]:
        """Get conversation history for a specific user"""
        return self.conversations.get(user_id, [])