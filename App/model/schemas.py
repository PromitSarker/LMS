from pydantic import BaseModel
from typing import List, Optional

# Quiz-related schemas
class PromptRequest(BaseModel):
    prompt: str
    language: Optional[str] = "english"  # Default to English if not specified

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str

class QuizResponse(BaseModel):
    quiz: List[QuizQuestion]

# Course-related schemas
class Assignment(BaseModel):
    title: str
    description: str
    expected_output: str

class TopicDetail(BaseModel):
    name: str
    explanation: str
    implementation: str
    troubleshooting: str
    exercises: List[str]
    assignment: Assignment

class Subtopic(BaseModel):
    title: str
    content_blocks: List[str]
    examples: List[str]
    practical_implications: List[str]

class ModuleContent(BaseModel):
    introduction: List[str]
    subtopics: List[Subtopic]
    consequences: List[str]

class Module(BaseModel):
    title: str
    content: ModuleContent
    learning_resources: List[str]

class CourseResponse(BaseModel):
    course_title: str
    course_description: List[str]
    learning_objectives: List[str]
    modules: List[Module]

class CourseRequest(BaseModel):
    topic: str
    difficulty: str = "intermediate"
    duration_weeks: int = 3

# Chat and Speech-related schemas
class ChatMessage(BaseModel):
    content: str

class ChatResponse(BaseModel):
    response: str

class SpeechToTextResponse(BaseModel):
    text: str
    detected_language: str

class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "en"

class AudioResponse(BaseModel):
    audio_path: str