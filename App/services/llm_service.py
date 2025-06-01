import logging
from fastapi import HTTPException
import httpx
from typing import Dict, List
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from App.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.base_url = settings.GROQ_API_URL
        self.model = settings.LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = httpx.Timeout(60.0, connect=10.0)
        self.max_retries = 3

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: logger.error(f"All retries failed: {retry_state.outcome}")
    )
    async def generate_completion(self, messages: List[Dict[str, str]], max_tokens: int = 4000) -> Dict:
        """
        Generate completion with automatic retries and improved error handling
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": settings.TEMPERATURE,
                    "max_tokens": max_tokens
                }
                
                logger.info(f"Sending request to Groq API with model: {self.model}")
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 429:
                    logger.warning("Rate limit hit, waiting before retry...")
                    await asyncio.sleep(2)
                    raise httpx.RequestError("Rate limit exceeded")
                
                if response.status_code != 200:
                    error_msg = f"Groq API error: {response.text}"
                    logger.error(error_msg)
                    if response.status_code == 401:
                        raise HTTPException(status_code=401, detail="Invalid API key")
                    elif response.status_code == 404:
                        raise HTTPException(status_code=404, detail="API endpoint not found")
                    else:
                        raise HTTPException(status_code=response.status_code, detail=error_msg)
                
                return response.json()
                
        except httpx.TimeoutException:
            logger.error("Request timed out")
            raise HTTPException(status_code=504, detail="Request timed out")
        except httpx.ConnectError:
            logger.error("Connection error")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API request error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")