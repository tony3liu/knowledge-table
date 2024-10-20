"""Service for interacting with OpenAI models."""

import logging
from typing import Any, List, Optional, Type

from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
import openai
from app.core.config import Settings
from app.services.llm.base import LLMService

logger = logging.getLogger(__name__)


class BaseLLMService(LLMService):
    """Base class for interacting with language models (OpenAI or Azure)."""

    def __init__(self, settings: Settings, client: Optional[openai] = None) -> None:
        self.settings = settings
        self.client = client  # Initialized by subclasses
        self.embeddings = None  # Will be initialized by subclasses

    def configure_openai(self, api_key: str) -> None:
        """Configure OpenAI with provided API key."""
        openai.api_key = api_key
        self.client = openai

    def configure_embeddings(self, model: str, api_key: str, api_base: Optional[str] = None) -> None:
        """Configure embeddings if needed."""
        self.embeddings = OpenAIEmbeddings(model=model, api_key=api_key, api_base=api_base)

    async def generate_completion(
        self, prompt: str, response_model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Generate a completion from the language model."""
        if self.client is None:
            logger.warning("Client is not initialized. Skipping generation.")
            return None

        try:
            response = self.client.ChatCompletion.create(
                engine=self.settings.llm_deployment,  # Can be overridden by subclasses
                messages=[{"role": "user", "content": prompt}]
            )

            parsed_response = response['choices'][0]['message']['content']
            logger.info(f"Generated response: {parsed_response}")

            if parsed_response is None:
                logger.warning("Received None response from the LLM")
                return None

            validated_response = response_model.parse_raw(parsed_response)
            if all(value is None for value in validated_response.dict().values()):
                logger.info("All fields in the response are None")
                return None

            return validated_response
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        if self.client is None:
            logger.warning("Client is not initialized. Skipping embeddings.")
            return []

        try:
            return self.embeddings.embed_documents(texts)
        except openai.OpenAIError as e:
            logger.error(f"Failed to retrieve embeddings: {e}")
            return []

    async def decompose_query(self, query: str) -> dict[str, Any]:
        """Decompose the query into smaller sub-queries."""
        if self.client is None:
            logger.warning("Client is not initialized. Skipping decomposition.")
            return {"sub_queries": [query]}

        return {"sub_queries": [query]}


class OpenAIService(BaseLLMService):
    """Service for interacting with OpenAI models."""

    def __init__(self, settings: Settings, client: Optional[openai] = None) -> None:
        super().__init__(settings, client)
        if not client:
            if settings.openai_api_key:
                self.configure_openai(settings.openai_api_key)
                self.configure_embeddings(settings.embedding_model, settings.openai_api_key)
            else:
                logger.warning("OpenAI API key is not set. LLM features will be disabled.")


class AzureOpenAIService(BaseLLMService):
    """Service for interacting with Azure OpenAI models."""

    def __init__(self, settings: Settings, client: Optional[openai] = None) -> None:
        super().__init__(settings, client)
        if not client:
            if settings.azure_openai_api_key and settings.azure_openai_api_base:
                openai.api_key = settings.azure_openai_api_key
                openai.api_base = settings.azure_openai_api_base
                openai.api_type = "azure"
                openai.api_version = settings.azure_openai_api_version

                self.configure_openai(settings.azure_openai_api_key)
                self.configure_embeddings(
                    settings.embedding_model,
                    settings.azure_openai_api_key,
                    settings.azure_openai_api_base
                )
            else:
                logger.warning("Azure OpenAI API key or base URL is not set. LLM features will be disabled.")
