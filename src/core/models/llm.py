from typing import List, Optional
from pydantic import Field
from dotenv import load_dotenv
import os
import asyncio

from tranformers import pipeline
from openai import OpenAI, AsyncOpenAI
import mlflow


from src.schema import Prompt, DomainObject, DomainObjectError

load_dotenv()

class LLM(DomainObject):
    """Base Multimodal Language Model that defines the interface for all language models"""

    def __init__(
        self,
        model_name: Optional[str] = Field(default="openai/gpt4o-mini", description="Name of the model in the form of a HuggingFace model name"),
        prompt: Optional[Prompt] = None,
        history: Optional[List[str]] = Field(default=[], description="History of messages"),
        stream: Optional[bool] = Field(default=False, description="Whether to stream the output")
        ):
        self.model_name = model_name
        self.prompt = prompt
        self.history = history
        self.stream = stream

        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.config = transformers.AutoConfig.from_pretrained(
            context.artifacts["snapshot"], trust_remote_code=True
        )



        mlflow.openai.autolog()

        provider = self.model_name.split("/")[0]

        if provider == "openai":
            API_KEY = os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "huggingface":
            API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
        elif provider == "google":
            os.environ.get("GEMINI_API_KEY")


        if self.stream:
            self.client = AsyncOpenAI(host=HOST, api_key=API_KEY)
        else:
            self.client = OpenAI(host=HOST, api_key=API_KEY)



        generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )

    async def __async_call__(self, prompt: Prompt):
        """Chat with the model"""
        self.prompt = prompt

        prompt.system_prompt



        await self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            temperature= prompt.temperature,
            max_tokens= prompt.max_tokens,
            top_p= prompt.top_p,
            frequency_penalty= prompt.frequency_penalty,
            presence_penalty= prompt.presence_penalty,

        max_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop_token: Optional[str] = None,
            top_p=
            max_tokens=

            response_model=request.response_model,
            stream=request.stream,
            messages=[
                {"role": "user", "content": f"Extract: `{data.query}`"},
            ],
        )

