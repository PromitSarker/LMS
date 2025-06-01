from App.core.config import settings
import json
import logging
from typing import Optional, Dict, Any, List
from fastapi import HTTPException
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class QuizGenerator:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.base_url = settings.GROQ_API_URL
        self.model = settings.LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def clean_json_string(self, json_str: str) -> str:
        """Clean and validate JSON string from LLM response."""
        try:
            # Find the first '[' and last ']' to extract valid JSON array
            start = json_str.find('[')
            end = json_str.rfind(']') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")
            return json_str[start:end]
        except Exception as e:
            logger.error(f"Error cleaning JSON string: {str(e)}")
            raise ValueError(f"Failed to clean JSON string: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_quiz(self, prompt: str, language: str = "english") -> Dict[str, List[Dict[str, Any]]]:
        """Generate a quiz with automatic retries and validation"""
        try:
            # Use supported languages from settings
            if language.lower() not in [lang.lower() for lang in settings.SUPPORTED_LANGUAGES]:
                logger.warning(f"Unsupported language {language}, falling back to English")
                language = "english"

            # Get language-specific system prompt from settings
            system_prompt = settings.SYSTEM_PROMPTS.get(language, settings.SYSTEM_PROMPTS["en"])
            
            base_prompt = (
                f"{system_prompt}\n"
                "You are an expert in OSHA regulations. "
                "Generate a multiple-choice quiz based on the following content. "
                "Follow these rules strictly:\n"
                "1. Create questions about OSHA safety procedures\n"
                "2. Each question must have EXACTLY four options (a, b, c, d)\n"
                "3. One and only one option should be correct\n"
                "4. Return ONLY a JSON array with this exact structure:\n"
                "[{\"question\": \"What is...\", \"options\": [\"a) option1\", \"b) option2\", \"c) option3\", \"d) option4\"], \"answer\": \"b) option2\"}, ...]\n"
                "5. Make sure the answer exactly matches one of the options\n"
                "6. Format all options with a), b), c), d) prefixes\n"
                "7. Return 5 questions unless specified otherwise in the prompt\n"
                "8. Do not include any explanations or additional text\n"
            )

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": base_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": settings.TEMPERATURE,
                "max_tokens": settings.MAX_TOKENS
            }

            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )

                if response.status_code == 429:
                    logger.warning("Rate limit hit, retrying...")
                    raise httpx.RequestError("Rate limit exceeded")

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Groq API error: {response.text}"
                    )

                result = response.json()["choices"][0]["message"]["content"].strip()
                logger.debug(f"Raw LLM output: {result}")

                # Clean and parse JSON
                cleaned_json = self.clean_json_string(result)
                quiz_data = json.loads(cleaned_json)

                # Validate and format questions
                formatted_quiz = []
                for i, question in enumerate(quiz_data, 1):
                    try:
                        self._validate_question(question, i)
                        formatted_quiz.append({
                            "question": f"Q{i}. {question['question']}",
                            "options": question["options"],
                            "answer": question["answer"]
                        })
                    except ValueError as e:
                        logger.warning(f"Question {i} validation failed: {str(e)}")
                        continue

                if not formatted_quiz:
                    raise ValueError("No valid questions were generated")

                return {"quiz": formatted_quiz}

        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {str(e)}\nRaw output: {result}")
            raise ValueError(f"Failed to parse quiz output: {str(e)}")
        except Exception as e:
            logger.error(f"Quiz generation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Quiz generation failed: {str(e)}"
            )

    def _validate_question(self, question: Dict, index: int) -> None:
        """Validate individual quiz question structure"""
        if not all(key in question for key in ["question", "options", "answer"]):
            raise ValueError(f"Question {index}: Missing required fields")

        if not isinstance(question["options"], list) or len(question["options"]) != 4:
            raise ValueError(f"Question {index}: Must have exactly 4 options")

        if question["answer"] not in question["options"]:
            raise ValueError(f"Question {index}: Answer must match one of the options")

        # Validate option prefixes
        prefixes = ["a)", "b)", "c)", "d)"]
        for i, option in enumerate(question["options"]):
            if not option.startswith(prefixes[i]):
                raise ValueError(f"Question {index}: Option {i+1} must start with {prefixes[i]}")

# Initialize singleton instance
quiz_generator = QuizGenerator()
