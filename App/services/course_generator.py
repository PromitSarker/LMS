from typing import Dict, List
import json
import re
import logging
from App.core.config import settings
from fastapi import HTTPException
from App.services.llm_service import LLMService
from App.model.schemas import CourseResponse

logger = logging.getLogger(__name__)

class CourseGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.default_language = settings.DEFAULT_LANGUAGE
        self.topics_by_week = {
            1: ["Fundamentals and Setup", "Core Concepts", "Basic Implementation"],
            2: ["Advanced Techniques", "Best Practices", "Performance Optimization"],
            3: ["Real-world Applications", "Production Deployment", "Monitoring and Maintenance"],
            4: ["Advanced Use Cases", "Scaling Solutions", "Future Trends"]
        }

    def _extract_json_from_text(self, text: str) -> Dict:
        try:
            text = text.strip()
            json_start = text.find('{')
            json_end = text.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")

            json_str = text[json_start:json_end]
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as je:
                if "Expecting ':' delimiter" in str(je):
                    json_str = re.sub(r'"\s+([{[\w])', '": \1', json_str)
                elif "Expecting value" in str(je):
                    json_str = re.sub(r',\s*$', '', json_str)
                    if not json_str.endswith('}'):
                        json_str += '}'
                return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {str(e)}\nText received: {text[:500]}...")
            raise ValueError(f"Could not extract valid JSON. Error: {str(e)}\nReceived:\n{text[:500]}")
        except Exception as e:
            logger.error(f"Unexpected error processing JSON: {str(e)}")
            raise ValueError(f"Error processing response: {str(e)}")

    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are an expert educational content creator."},
            {"role": "user", "content": prompt}
        ]

    async def generate_course(self, topic: str,  language: str = None) -> CourseResponse:
        try:
            if language and language not in settings.SUPPORTED_LANGUAGES:
                logger.warning(f"Unsupported language: {language}. Falling back to {self.default_language}")
                language = self.default_language

            prompt = self._create_prompt(topic, language)
            messages = self._create_messages(prompt)
            response = await self.llm_service.generate_completion(messages)
            content = response['choices'][0]['message']['content']
            logger.info("LLM raw output:\n%s", content)

            course_data = self._extract_json_from_text(content)
            self._validate_course_structure(course_data)
            return CourseResponse(**course_data)

        except Exception as e:
            logger.error(f"Course generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Course generation failed: {str(e)}")

    def _validate_course_structure(self, course_data: Dict) -> None:
        required_top_fields = ["course_title", "course_description", "learning_objectives", "modules"]
        for field in required_top_fields:
            if field not in course_data:
                raise ValueError(f"Missing required top-level field: {field}")

        if not isinstance(course_data["modules"], list) or not course_data["modules"]:
            raise ValueError("'modules' must be a non-empty list")

        for i, module in enumerate(course_data["modules"]):
            if not all(key in module for key in ["title", "content", "learning_resources"]):
                raise ValueError(f"Module {i} missing required fields")

            content = module["content"]
            if not all(key in content for key in ["introduction", "subtopics", "consequences"]):
                raise ValueError(f"Module {i} content missing required fields")

            for j, subtopic in enumerate(content["subtopics"]):
                if not all(key in subtopic for key in ["title", "content_blocks", "examples"]):
                    raise ValueError(f"Module {i} subtopic {j} missing required fields")

    def _create_prompt(self, topic: str,  language: str = None) -> str:
        lang = language if language in settings.SUPPORTED_LANGUAGES else self.default_language

        language_names = {
            "en": "English", "es": "Spanish", "fr": "French", "ko": "Korean", "fa": "Persian",
            "vi": "Vietnamese", "tl": "Tagalog", "de": "German", "zh": "Chinese", "bn": "Bengali", "hi": "Hindi"
        }

        base_prompts = {
            "en": f"Create a 3-week intermediate level course on {topic}.",
            "es": f"Crea un curso de 3 semanas de nivel intermediate sobre {topic}.",
            "fr": f"Créez un cours de 3 semaines de niveau intermediate sur {topic}.",
            "de": f"Erstellen Sie einen 3-wöchigen intermediate Kurs über {topic}.",
            "zh": f"创建一个关于{topic}的intermediate级别的3周课程。",
            "bn": f"{topic} এর উপর intermediate স্তরের 3 সপ্তাহের কোর্স তৈরি করুন। সমস্ত কনটেন্ট বাংলায় তৈরি করুন।",
            "hi": f"{topic} पर intermediate स्तर का 3 सप्ताह का पाठ्यक्रम बनाएं। सभी सामग्री हिंदी में उत्पन्न करें।",
            "ko": f"{topic}에 대한 intermediate 수준의 3주 과정 생성. 모든 콘텐츠를 한국어로 생성하세요.",
            "fa": f"{topic} در سطح intermediate دوره 3 هفته ای ایجاد کنید. تمام محتوا را به فارسی تولید کنید.",
            "vi": f"Tạo khóa học intermediate 3 tuần về {topic}. Tạo TẤT CẢ nội dung bằng tiếng Việt.",
            "tl": f"Lumikha ng intermediate na antas na 3 linggong kurso sa {topic}. Lumikha ng LAHAT ng nilalaman sa Tagalog."
        }

        system_instructions = {
            "en": "You MUST generate ALL content in English.",
            "es": "DEBES generar TODO el contenido en español.",
            "fr": "Vous DEVEZ générer TOUT le contenu en français.",
            "de": "Sie MÜSSEN ALLE Inhalte auf Deutsch generieren.",
            "zh": "您必须用中文生成所有内容。",
            "bn": "আপনাকে অবশ্যই সমস্ত কনটেন্ট বাংলায় তৈরি করতে হবে।",
            "hi": "आपको सभी सामग्री हिंदी में उत्पन्न करनी चाहिए।",
            "ko": "모든 콘텐츠를 한국어로 생성해야 합니다.",
            "fa": "شما باید تمام محتوا را به فارسی تولید کنید.",
            "vi": "Bạn PHẢI tạo TẤT CẢ nội dung bằng tiếng Việt.",
            "tl": "KAILANGAN mong lumikha ng LAHAT ng nilalaman sa Tagalog."
        }

        prompt_template = f'''{base_prompts[lang]}

{system_instructions[lang]}

Follow this structure exactly. Return ONLY valid JSON like this:

{{
  "course_title": "Professional title of course",
  "course_description": ["Paragraph 1", "Paragraph 2"],
  "learning_objectives": ["Objective 1", "Objective 2"],
  "modules": [
    {{
      "title": "Module title",
      "content": {{
        "introduction": ["Intro paragraph 1", "Intro paragraph 2"],
        "subtopics": [
          {{
            "title": "Subtopic title",
            "content_blocks": ["Paragraph 1", "Paragraph 2", "Paragraph 3"],
            "examples": ["Example 1", "Example 2"],
            "practical_implications": ["Impact 1", "Impact 2"]
          }}
        ],
        "consequences": ["Paragraph 1", "Paragraph 2", "Paragraph 3"]
      }},
      "learning_resources": ["Resource 1", "Resource 2"]
    }}
  ]
}}'''

        return prompt_template
