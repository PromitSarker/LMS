import os
import logging
from pydantic import BaseModel, validator
from dotenv import load_dotenv
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    # Project settings
    PROJECT_NAME: str = "AI Based LMS"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = "A LMS that uses AI to generate course content and provide personalized learning experiences."
    
    # API settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Updated to a more capable model
    # Rephrasing settings
    MAX_TOKENS: int = max
    TEMPERATURE: float = 0.6  # Reduced temperature for more consistent JSON output
    
    # Language settings
    SUPPORTED_LANGUAGES: List[str] = ["en", "es", "fr", "ko", "fa", "vi", "tl", "de", "zh", "bn", "hi"]
    DEFAULT_LANGUAGE: str = "en"
    
    # Language-specific prompts
    SYSTEM_PROMPTS: Dict[str, str] = {
        "en": "You are a helpful AI assistant. Provide clear, concise, and accurate responses.",
        "es": "Eres un asistente de IA útil. Proporciona respuestas claras, concisas y precisas.",
        "fa": "شما یک دستیار هوش مصنوعی مفید هستید. لطفاً پاسخ‌های واضح، مختصر و دقیق ارائه دهید.",
        "vi": "Bạn là một trợ lý AI hữu ích. Hãy cung cấp câu trả lời rõ ràng, ngắn gọn và chính xác.",
        "ko": "당신은 유용한 AI 도우미입니다. 명확하고 간결하며 정확한 답변을 제공해주세요.",
        "tl": "Ikaw ay isang kapaki-pakinabang na AI na katulong. Mangyaring magbigay ng malinaw, maikli, at tamang mga sagot.",
        "fr": "Vous êtes un assistant IA utile. Fournissez des réponses claires, concises et précises.",
        "de": "Sie sind ein hilfreicher KI-Assistent. Geben Sie klare, präzise und akkurate Antworten.",
        "zh": "你是一个有帮助的AI助手。提供清晰、简洁和准确的回答。",
        "bn": "আপনি একটি সহায়ক AI সহায়ক। পরিষ্কার এবং সঠিক প্রতিক্রিয়া দিন।",
        "hi": "आप एक सहायक एआई सहायक हैं। स्पष्ट, संक्षिप्त और सटीक उत्तर प्रदान करें।",
    }
    
    # Speech settings
    AUDIO_FILE_DIR: str = "audio_files"
    MAX_AUDIO_LENGTH: int = 300  # Maximum audio length in seconds
    SPEECH_TO_TEXT_MODEL: str = "llama-3.3-70b-versatile"  # Use same model as LLM for transcription
    SPEECH_MAX_TOKENS: int = 1000
    SPEECH_TEMPERATURE: float = 0.3  # Lower temperature for more accurate transcription
    
    # Connection settings
    REQUEST_TIMEOUT: int = 60
    CONNECT_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1
    MAX_RETRY_DELAY: int = 10
    
    # Health check settings
    HEALTH_CHECK_INTERVAL: int = 60  # seconds
    SERVICE_DEGRADED_THRESHOLD: float = 0.8  # 80% success rate threshold
    
    @validator('GROQ_API_KEY')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("GROQ_API_KEY environment variable must be set")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Initialize settings with error handling
try:
    settings = Settings()
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.error(f"Failed to load settings: {str(e)}")
    raise