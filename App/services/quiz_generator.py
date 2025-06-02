from typing import Dict, List, Any
import json
import logging
from fastapi import HTTPException
from App.core.config import settings
from App.model.schemas import CourseResponse, QuizResponse, QuizQuestion
from App.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class QuizGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.default_language = settings.DEFAULT_LANGUAGE

    def _create_messages(self, course_content: CourseResponse, language: str = None) -> List[Dict[str, str]]:
        """
        Create a message array for the LLM API call to generate quiz questions
        """
        prompt = self._create_prompt(course_content, language)
        return [
            {"role": "system", "content": "You are an expert educational content creator specializing in creating assessment questions."},
            {"role": "user", "content": prompt}
        ]

    def _create_prompt(self, course_content: CourseResponse, language: str = None) -> str:
        """
        Create a prompt for quiz generation based on course content
        """
        lang = language if language in settings.SUPPORTED_LANGUAGES else self.default_language
        
        # Format course content for the prompt
        course_text = f"""
Course Title: {course_content.course_title}
Course Description: {course_content.course_description}

Learning Objectives:
{chr(10).join(f"- {obj}" for obj in course_content.learning_objectives)}

Course Modules:
{chr(10).join(self._format_module(module, i) for i, module in enumerate(course_content.modules, 1))}
"""

        base_prompt = {
            "en": f'''Based on the following course content, generate a comprehensive quiz that tests understanding of key concepts:

{course_text}

Generate a quiz with the following structure:
1. Multiple choice questions (at least 3)
2. True/False questions (at least 2)
3. Short answer questions (at least 2)
4. Practical application questions (at least 1)

For each question, include:
- The question text
- Correct answer
- Explanation of the correct answer
- Difficulty level (Beginner/Intermediate/Advanced)

Return ONLY a valid JSON array with this exact structure:
[
    {{
        "question": "The actual question",
        "question_type": "multiple_choice/true_false/short_answer/practical",
        "options": ["Option 1", "Option 2", "Option 3", "Option 4"],  # Only for multiple choice
        "answer": "The correct answer",
        "explanation": "Detailed explanation of why this is correct",
        "difficulty": "Beginner/Intermediate/Advanced"
    }}
]''',
            "es": f'''Basado en el siguiente contenido del curso, genera un cuestionario completo que pruebe la comprensión de los conceptos clave:

{course_text}

Genera un cuestionario con la siguiente estructura:
1. Preguntas de opción múltiple (mínimo 3)
2. Preguntas Verdadero/Falso (mínimo 2)
3. Preguntas de respuesta corta (mínimo 2)
4. Preguntas de aplicación práctica (mínimo 1)

Para cada pregunta, incluye:
- El texto de la pregunta
- Respuesta correcta
- Explicación de la respuesta correcta
- Nivel de dificultad (Principiante/Intermedio/Avanzado)

Devuelve SOLO un array JSON válido con esta estructura exacta:
[
    {{
        "question": "La pregunta actual",
        "question_type": "multiple_choice/true_false/short_answer/practical",
        "options": ["Opción 1", "Opción 2", "Opción 3", "Opción 4"],  # Solo para opción múltiple
        "answer": "La respuesta correcta",
        "explanation": "Explicación detallada de por qué esto es correcto",
        "difficulty": "Principiante/Intermedio/Avanzado"
    }}
]''',
            # Add other language prompts similar to course_generator.py
        }[lang]

        return base_prompt

    def _format_module(self, module: Any, index: int) -> str:
        """Format a module's content for the prompt"""
        topics_text = []
        for topic in module.topics:
            topic_text = f"""
            Topic: {topic.name}
            Explanation: {topic.explanation}
            Implementation: {topic.implementation}
            Troubleshooting: {topic.troubleshooting}
            """
            topics_text.append(topic_text)

        # Use getattr to safely access overview, defaulting to empty string if not present
        overview = getattr(module, 'overview', '')
        
        return f"""
        Module {index}: {module.title}
        Overview: {overview}
        Topics:
        {chr(10).join(topics_text)}
        Teaching Resources:
        {chr(10).join(f"- {resource}" for resource in module.teaching_resources)}
        """

    async def generate_quiz(self, course_content: CourseResponse, language: str = None) -> QuizResponse:
        """
        Generate a quiz based on the provided course content
        """
        try:
            messages = self._create_messages(course_content, language)
            response = await self.llm_service.generate_completion(messages)
            content = response['choices'][0]['message']['content']
            
            # Extract and validate JSON from response
            quiz_data = self._extract_json_from_text(content)
            
            # Validate quiz structure
            self._validate_quiz_structure(quiz_data)
            
            # Convert to QuizResponse model
            formatted_questions = self._format_questions(quiz_data)
            quiz = QuizResponse(quiz=formatted_questions)
            
            return quiz

        except Exception as e:
            logger.error(f"Quiz generation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Quiz generation failed: {str(e)}"
            )

    def _extract_json_from_text(self, text: str) -> List[Dict]:
        """
        Extracts and validates JSON from LLM response
        """
        try:
            # Clean the text first
            text = text.strip()
            
            # Try to find complete JSON array
            json_start = text.find('[')
            json_end = text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = text[json_start:json_end]
            return json.loads(json_str)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {str(e)}\nText received: {text[:500]}...")
            raise ValueError(f"Could not extract valid JSON. Error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing JSON: {str(e)}")
            raise ValueError(f"Error processing response: {str(e)}")

    def _validate_quiz_structure(self, quiz_data: List[Dict]) -> None:
        """
        Validates that the quiz data contains all required fields
        """
        if not isinstance(quiz_data, list) or not quiz_data:
            raise ValueError("Quiz data must be a non-empty list")
            
        for i, question in enumerate(quiz_data):
            required_fields = ["question", "question_type", "answer", "explanation", "difficulty"]
            for field in required_fields:
                if field not in question:
                    raise ValueError(f"Question {i} is missing required field: {field}")
            
            if question["question_type"] == "multiple_choice":
                if "options" not in question or not isinstance(question["options"], list) or len(question["options"]) != 4:
                    raise ValueError(f"Multiple choice question {i} must have exactly 4 options")

    def _format_questions(self, quiz_data: List[Dict]) -> List[QuizQuestion]:
        """
        Format the quiz questions to match the QuizQuestion model
        """
        formatted_questions = []
        for i, question in enumerate(quiz_data, 1):
            formatted_question = {
                "question": f"Q{i}. {question['question']}",
                "options": question.get("options", []),
                "answer": question["answer"]
            }
            formatted_questions.append(QuizQuestion(**formatted_question))
        return formatted_questions

# Initialize singleton instance with LLMService
from App.services.llm_service import LLMService
llm_service = LLMService()  # Create LLMService instance
quiz_generator = QuizGenerator(llm_service=llm_service)  # Pass llm_service to QuizGenerator