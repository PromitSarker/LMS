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
        self.model = settings.SPEECH_TO_TEXT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Update model reference
        self.speech_model = settings.SPEECH_TO_TEXT_MODEL
        self.max_audio_duration = 30  # maximum duration in seconds
        self.target_sample_rate = 16000  # lower sample rate for smaller file size

        self.course_generator_url = "http://localhost:8000/generate-course"  # Add base URL for course generator

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def text_to_speech(self, text: str, language: str = None) -> str:
        """Convert text to speech with retry logic"""
        try:
            lang = language if language in settings.SUPPORTED_LANGUAGES else settings.DEFAULT_LANGUAGE
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
    async def speech_to_text(self, audio_file: UploadFile) -> Tuple[str, str]:
        """Convert speech to text using Groq AI with audio size optimization"""
        try:
            content = await audio_file.read()
            
            # Preprocess audio to reduce size
            processed_audio = await self.preprocess_audio(content)
            
            # Convert to base64 with size check
            audio_base64 = base64.b64encode(processed_audio).decode('utf-8')
            
            # Estimate token size (rough estimation)
            estimated_tokens = len(audio_base64) / 3  # approximate tokens per base64 chunk
            
            if estimated_tokens > settings.SPEECH_MAX_TOKENS:
                raise HTTPException(
                    status_code=413,
                    detail="Audio file too large. Please limit recording to 30 seconds or less."
                )

            # Create a prompt for transcription
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at transcribing audio. Extract the speech content accurately."
                },
                {
                    "role": "user",
                    "content": f"Transcribe this audio: {audio_base64}"
                }
            ]

            # Make request to Groq AI with increased timeout
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.speech_model,
                        "messages": messages,
                        "temperature": settings.SPEECH_TEMPERATURE,
                        "max_tokens": settings.SPEECH_MAX_TOKENS
                    }
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Groq API error: {response.text}"
                    )

                result = response.json()
                transcription = result["choices"][0]["message"]["content"]

                # Detect language
                lang_messages = [
                    {
                        "role": "system",
                        "content": "You are a language detection expert. Reply only with the ISO 639-1 language code."
                    },
                    {
                        "role": "user",
                        "content": f"What is the language code for this text: {transcription}"
                    }
                ]

                lang_response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.speech_model,
                        "messages": lang_messages,
                        "temperature": 0.1,
                        "max_tokens": 2
                    }
                )

                if lang_response.status_code != 200:
                    detected_lang = settings.DEFAULT_LANGUAGE
                else:
                    detected_lang = lang_response.json()["choices"][0]["message"]["content"].strip()

                logger.info(f"Transcribed audio to text. Detected language: {detected_lang}")
                return transcription.strip(), detected_lang

        except HTTPException as he:
            raise he
        except Exception as e:
            error_msg = f"Speech-to-text conversion failed: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def record_audio(self, duration: int = 5) -> Optional[str]:
        """Record audio from microphone"""
        try:
            with sr.Microphone() as source:
                logger.info("Recording audio...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=duration)
                
                # Save audio to recordings directory
                filename = f"recording_{int(time.time())}.wav"
                filepath = self.recordings_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(audio.get_wav_data())
                    
                logger.info(f"Saved recording to {filepath}")
                return str(filepath)
                    
        except Exception as e:
            error_msg = f"Audio recording failed: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

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
            module_filename = f"module_{module_idx}_{self.sanitize_filename(title)}.mp3"
            module_filepath = self.tts_dir / module_filename

            if not module_filepath.exists():
                logger.info(f"Generating audio for module {module_idx}: {title}")
                audio_path = await self.text_to_speech(module_content, language)
                return audio_path
            
            return str(module_filepath)

        except Exception as e:
            error_msg = f"Failed to generate module audio: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def create_course_audio(self, course_response: CourseResponse, language: str = None) -> Dict[str, Any]:
        """Create separate audio files for each module in the course"""
        try:
            # Format course introduction
            intro_text = f"""Welcome to {course_response.course_title}.
            
Course Description:
{'. '.join(course_response.course_description)}

Learning Objectives:
{'. '.join(f'Objective {i+1}: {obj}' for i, obj in enumerate(course_response.learning_objectives, 1))}
"""
            # Generate intro audio
            intro_audio_path = await self.text_to_speech(intro_text, language)
            
            # Generate audio for each module
            module_audios = []
            for idx, module in enumerate(course_response.modules, 1):
                # Format module content
                module_text = await self._format_module_content(idx, module)
                
                # Generate audio for this module
                module_audio_path = await self.text_to_speech(module_text, language)
                
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