import os
import logging
import tempfile
import speech_recognition as sr
from typing import Tuple, Optional, List, Dict, Any
from fastapi import HTTPException, UploadFile
import asyncio
import base64
import httpx
from pathlib import Path
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from App.core.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential
from gtts import gTTS
import re

from App.model.schemas import CourseResponse

logger = logging.getLogger(__name__)

class SpeechService:
    def __init__(self):
        # Create structured directories for different audio types
        self.base_dir = Path("App/static/audio")
        self.tts_dir = self.base_dir / "tts"
        self.recordings_dir = self.base_dir / "recordings"
        self.temp_dir = self.base_dir / "temp"
        
        # Create all directories
        for directory in [self.tts_dir, self.recordings_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.recognizer = sr.Recognizer()
        self.api_key = settings.GROQ_API_KEY
        self.base_url = settings.GROQ_API_URL

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Update model reference
        self.max_audio_duration = 30  # maximum duration in seconds
        self.target_sample_rate = 16000  # lower sample rate for smaller file size

        self.course_generator_url = "http://localhost:8000/generate-course"  # Add base URL for course generator

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def text_to_speech(self, text: str, language: str = None, custom_filename: str = None) -> str:
        """Convert text to speech with retry logic"""
        try:
            lang = language if language in settings.SUPPORTED_LANGUAGES else settings.DEFAULT_LANGUAGE
            
            # Use custom filename if provided, otherwise use hash
            if custom_filename:
                filename = f"{custom_filename}_{lang}.mp3"
            else:
                filename = f"tts_{hash(text)}_{lang}.mp3"
                
            filepath = self.tts_dir / filename
            
            if not filepath.exists():
                tts = gTTS(text=text, lang=lang, slow=False)
                try:
                    tts.save(str(filepath))
                    logger.info(f"Generated audio file: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to save audio file: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to generate audio file: {str(e)}"
                    )
                
            return str(filepath)
            
        except Exception as e:
            error_msg = f"Text-to-speech conversion failed: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def preprocess_audio(self, audio_content: bytes) -> bytes:
        """Preprocess audio to reduce file size"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
                temp_in.write(audio_content)
                temp_in_path = temp_in.name

            # Convert to mono and resample
            audio = AudioSegment.from_wav(temp_in_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(self.target_sample_rate)
            
            # Compress audio quality
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
                audio.export(temp_out.name, format='wav', parameters=["-q:a", "0"])
                with open(temp_out.name, 'rb') as f:
                    processed_audio = f.read()

            # Cleanup temp files
            os.unlink(temp_in_path)
            os.unlink(temp_out.name)

            return processed_audio

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )

    def sanitize_filename(self, text: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', text.strip().replace(' ', '_'))[:50]

    async def narrate_module(self, module_idx: int, module: 'Module', language: str = None) -> Dict[str, Any]:
        """Narrate a single module and combine its content into one audio file"""
        try:
            # Format the module content in a structured way
            module_content = f"""
Module {module_idx}: {module.title}

Introduction:
{'. '.join(module.content.introduction)}

"""
            # Add subtopics
            for sub_idx, subtopic in enumerate(module.content.subtopics, 1):
                module_content += f"""
Subtopic {sub_idx}: {subtopic.title}

Content:
{'. '.join(subtopic.content_blocks)}

Examples:
{'. '.join(f'Example {i+1}: {ex}' for i, ex in enumerate(subtopic.examples))}

Practical Implications:
{'. '.join(f'Implication {i+1}: {imp}' for i, imp in enumerate(subtopic.practical_implications))}
"""

            # Add consequences
            module_content += f"""
Consequences:
{'. '.join(f'Consequence {i+1}: {cons}' for i, cons in enumerate(module.content.consequences))}

Learning Resources:
{'. '.join(f'Resource {i+1}: {res}' for i, res in enumerate(module.learning_resources))}
"""

            # Generate single audio file for the module
            module_filename = f"module_{module_idx}_{self.sanitize_filename(module.title)}.mp3"
            module_filepath = self.tts_dir / module_filename
            
            # Generate audio if it doesn't exist
            if not module_filepath.exists():
                logger.info(f"Generating audio for module {module_idx}: {module.title}")
                logger.debug(f"Module content to narrate: {module_content}")
                
                audio_path = await self.text_to_speech(module_content, language)
                logger.info(f"Generated audio file for module {module_idx}: {audio_path}")
                
                return {
                    "module_number": module_idx,
                    "title": module.title,
                    "audio_file": audio_path
                }
            else:
                logger.info(f"Using existing audio file for module {module_idx}: {module_filepath}")
                return {
                    "module_number": module_idx,
                    "title": module.title,
                    "audio_file": str(module_filepath)
                }

        except Exception as e:
            error_msg = f"Failed to narrate module {module_idx}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def narrate_course(self, course: 'CourseResponse', language: str = None) -> Dict[str, Any]:
        """
        Narrates a complete course by creating separate audio files for course info and each module.
        """
        try:
            # Format course introduction content
            course_intro = f"""
Welcome to {course.course_title}

Course Description:
{'. '.join(course.course_description)}

Learning Objectives:
{'. '.join(f'Objective {i+1}: {obj}' for i, obj in enumerate(course.learning_objectives))}
"""

            course_content = {
                "course_info": {
                    "title": await self.text_to_speech(course_intro, language)
                },
                "modules": []
            }

            # Process each module separately
            for module_idx, module in enumerate(course.modules, 1):
                module_audio = await self.narrate_module(module_idx, module, language)
                course_content["modules"].append(module_audio)
                logger.info(f"Completed narration for module {module_idx}: {module.title}")

            total_files = len(course_content["modules"]) + 1
            logger.info(f"Course narration completed. Generated {total_files} audio files.")
            
            return course_content

        except Exception as e:
            error_msg = f"Course narration failed: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def fetch_course_content(self, topic: str, difficulty: str = "intermediate", 
                             duration_weeks: int = 3, language: str = None) -> CourseResponse:
        """Fetch course content directly from the course generator endpoint"""
        try:
            request_data = CourseRequest(
                topic=topic,
                difficulty=difficulty,
                duration_weeks=duration_weeks
            )
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.course_generator_url}",
                    params={"language": language or settings.DEFAULT_LANGUAGE},
                    json=request_data.dict()
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Failed to fetch course content: {response.text}"
                    )

                return CourseResponse(**response.json())

        except Exception as e:
            error_msg = f"Failed to fetch course content: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def generate_module_audio(self, module_idx: int, module_content: str, 
                                  title: str, language: str = None) -> str:
        """Generate audio file for a single module"""
        try:
            # Create standardized filename
            safe_title = self.sanitize_filename(title)
            module_filename = f"module_{module_idx}_{safe_title}"
            
            # Generate audio with custom filename
            audio_path = await self.text_to_speech(
                text=module_content,
                language=language,
                custom_filename=module_filename
            )
            
            return audio_path

        except Exception as e:
            error_msg = f"Failed to generate module audio: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def create_course_audio(self, course_response: CourseResponse, language: str = None) -> Dict[str, Any]:
        """Create separate audio files for each module in the course"""
        try:
            # Format and generate intro audio
            intro_text = f"""Welcome to {course_response.course_title}.
            
Course Description:
{'. '.join(course_response.course_description)}

Learning Objectives:
{'. '.join(f'Objective {i+1}: {obj}' for i, obj in enumerate(course_response.learning_objectives, 1))}
"""
            intro_filename = f"intro_{self.sanitize_filename(course_response.course_title)}"
            intro_audio_path = await self.text_to_speech(
                text=intro_text,
                language=language,
                custom_filename=intro_filename
            )
            
            # Generate audio for each module
            module_audios = []
            for idx, module in enumerate(course_response.modules, 1):
                module_text = await self._format_module_content(idx, module)
                module_filename = f"module_{idx}_{self.sanitize_filename(module.title)}"
                
                module_audio_path = await self.text_to_speech(
                    text=module_text,
                    language=language,
                    custom_filename=module_filename
                )
                
                module_audios.append({
                    "module_number": idx,
                    "title": module.title,
                    "audio_file": str(module_audio_path),
                    "duration": await self._get_audio_duration(module_audio_path)
                })
            
            return {
                "course_title": course_response.course_title,
                "audio_files": {
                    "introduction": {
                        "title": "Course Introduction",
                        "audio_file": str(intro_audio_path),
                        "duration": await self._get_audio_duration(intro_audio_path)
                    },
                    "modules": module_audios
                },
                "total_modules": len(module_audios),
                "language": language or settings.DEFAULT_LANGUAGE
            }

        except Exception as e:
            error_msg = f"Failed to create course audio: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def _format_module_content(self, idx: int, module: 'Module') -> str:
        """Helper method to format module content for narration"""
        module_text = f"""
Module {idx}: {module.title}

Introduction:
{'. '.join(module.content.introduction)}

"""
        # Add subtopics
        for sub_idx, subtopic in enumerate(module.content.subtopics, 1):
            module_text += f"""
Subtopic {sub_idx}: {subtopic.title}

Content:
{'. '.join(subtopic.content_blocks)}

Examples:
{'. '.join(f'Example {i+1}: {ex}' for i, ex in enumerate(subtopic.examples, 1))}

Practical Implications:
{'. '.join(f'Implication {i+1}: {imp}' for i, imp in enumerate(subtopic.practical_implications, 1))}
"""

        # Add consequences and resources
        module_text += f"""
Key Takeaways:
{'. '.join(module.content.consequences)}

Learning Resources:
{'. '.join(f'Resource {i+1}: {res}' for i, res in enumerate(module.learning_resources, 1))}
"""
        return module_text

    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get the duration of an audio file in seconds"""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            logger.warning(f"Could not get audio duration: {str(e)}")
            return 0.0