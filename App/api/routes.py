from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
import os  # Add this import
from App.services.quiz_generator import quiz_generator  # Import singleton instance
from App.model.schemas import PromptRequest, QuizResponse
from App.model.schemas import (
    CourseRequest, 
    CourseResponse, 
    ChatMessage, 
    ChatResponse, 
    TextToSpeechRequest
)
from App.services.course_generator import CourseGenerator
from App.services.llm_service import LLMService
from App.services.chat_service import ChatService
from App.services.speech_service import SpeechService
from App.core.config import settings
import json
import httpx
from datetime import datetime
from pathlib import Path
import logging

router = APIRouter()

# Initialize services
llm_service = LLMService()
course_generator = CourseGenerator(llm_service)
speech_service = SpeechService()
chat_service = ChatService()

logger = logging.getLogger(__name__)

@router.post("/Course-Generation/")
async def narrate_course(
    request: CourseRequest,
    language: str = Query(default=settings.DEFAULT_LANGUAGE, enum=settings.SUPPORTED_LANGUAGES)
):
    """
    Generate audio files for each module and return both audio content and JSON course data.
    """
    try:
        # Generate course content directly using course generator
        course_response = await course_generator.generate_course(
            topic=request.topic,
            difficulty=request.difficulty,
            duration_weeks=request.duration_weeks,
            language=language
        )

        # Generate audio content for each module
        audio_content = await speech_service.create_course_audio(
            course_response=course_response,
            language=language
        )

        # Prepare detailed JSON response
        return {
            "course_content": {
                "title": course_response.course_title,
                "description": course_response.course_description,
                "learning_objectives": course_response.learning_objectives,
                "modules": [
                    {
                        "title": module.title,
                        "content": {
                            "introduction": module.content.introduction,
                            "subtopics": [
                                {
                                    "title": subtopic.title,
                                    "content_blocks": subtopic.content_blocks,
                                    "examples": subtopic.examples,
                                    "practical_implications": subtopic.practical_implications
                                }
                                for subtopic in module.content.subtopics
                            ],
                            "consequences": module.content.consequences,
                            "learning_resources": module.learning_resources
                        }
                    }
                    for module in course_response.modules
                ]
            },
            "audio_content": {
                "course_title": course_response.course_title,
                "modules_audio": audio_content,
                "language": language,
                "total_modules": len(course_response.modules)
            }
        }

    except Exception as e:
        logger.error(f"Course narration failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate and narrate course: {str(e)}"
        )


@router.post("/generate_quiz/", response_model=QuizResponse)
async def create_quiz(prompt_request: PromptRequest):
    """Generate a quiz from text prompt"""
    try:
        quiz_response = await quiz_generator.generate_quiz(
            prompt=prompt_request.prompt,
            language=prompt_request.language
        )
        return quiz_response
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.post("/chat/", response_model=ChatResponse)
async def chat_with_ai(
    message: ChatMessage,
    user_id: str = Query(default="default", description="User identifier for conversation tracking"),
    voice_response: bool = Query(default=False, description="Return audio response"),
    language: str = Query(default=settings.DEFAULT_LANGUAGE, enum=settings.SUPPORTED_LANGUAGES)
):
    """Chat endpoint with conversation memory and optional voice response"""
    response_text = await chat_service.get_chat_response(
        message.content,
        user_id=user_id,
        language=language
    )
    
    if voice_response:
        audio_path = await speech_service.text_to_speech(response_text, language)
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename=os.path.basename(audio_path)
        )
    
    return ChatResponse(response=response_text)

@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to AI Course Generator API"}

@router.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check LLM service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(settings.GROQ_API_URL.replace("/chat/completions", ""))
            health_status["services"]["llm"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        health_status["services"]["llm"] = f"unhealthy: {str(e)}"

    # Check file system for audio storage
    try:
        audio_path = Path("App/static/audio")
        if audio_path.exists() and audio_path.is_dir():
            health_status["services"]["file_system"] = "healthy"
        else:
            health_status["services"]["file_system"] = "unhealthy: audio directory not found"
    except Exception as e:
        health_status["services"]["file_system"] = f"unhealthy: {str(e)}"

    # Overall status
    health_status["status"] = "healthy" if all(
        status == "healthy" for status in health_status["services"].values()
    ) else "unhealthy"
    
    return health_status

@router.get("/ping")
async def ping():
    """Simple ping endpoint for load balancers"""
    return {"status": "ok"}


@router.delete("/chat/{user_id}/history")
async def clear_chat_history(user_id: str):
    """Clear chat history for a specific user"""
    chat_service.clear_history(user_id)
    return {"message": f"Chat history cleared for user {user_id}"}

@router.get("/chat/{user_id}/history")
async def get_chat_history(user_id: str):
    """Get chat history for a specific user"""
    return {"history": chat_service.get_history(user_id)}

