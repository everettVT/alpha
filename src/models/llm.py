from typing import List, Optional
from pydantic import Field
from dotenv import load_dotenv
import os

from tranformers import pipeline
from openai import OpenAI, AsyncOpenAI
import mlflow


from src.schema import Prompt, DomainObject, DomainObjectError,

load_dotenv()

await self.client.chat.completions.create(
        model=self.model_name,
        response_model=request.response_model,
        stream=request.stream,
        messages=[
            {"role": "user", "content": f"Extract: `{data.query}`"},
        ],
    )

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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts["snapshot"], padding_side="left"
        )

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts["snapshot"], trust_remote_code=True
        )

        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        HOST = os.environ.get("HOST")



        if self.stream:
            self.client = AsyncOpenAI(host=HOST, api_key="sk-test")
        else:
            self.client = OpenAI(host=HOST, api_key="sk-test")



        generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )

    async def __async_call__(self, request: Request):
        """Chat with the model"""
        self.prompt =
        self.chat(self.prompt)

    @abstractmethod
    def chat(self, prompt: str, history: str = "") -> str:
        """Chat with the model"""

    
    @abstractmethod
    def complete(self, prompt: str) -> str: 
        """Request Single Prompt Completion"""

    def set_temperature(self, value: float):
        """Set Temperature"""
        self.temperature = value

    def set_max_tokens(self, value: int):
        """Set new max tokens"""
        self.max_tokens = value

    def clear_history(self):
        """Clear history"""
        self.history = []

    def enable_logging(self, log_file: str = "model.log"):
        """Initialize logging for the model."""
        logging.basicConfig(filename=log_file, level=logging.INFO)
        self.log_file = log_file

    def log_event(self, message: str):
        """Log an event."""
        logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    def save_checkpoint(self, checkpoint_dir: str = "checkpoints"):
        """Save the model state."""
        # This is a placeholder for actual checkpointing logic.
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_{time.strftime("%Y%m%d-%H%M%S")}.ckpt',
        )
        # Save model state to checkpoint_path
        self.log_event(f"Model checkpoint saved at {checkpoint_path}")

    def set_max_length(self, max_length: int):
        """Set max length

        Args:
            max_length (int): _description_
        """
        self.max_length = max_length

    def set_model_name(self, model_name: str):
        """Set model name

        Args:
            model_name (str): _description_
        """
        self.model_name = model_name

    def set_frequency_penalty(self, frequency_penalty: float):
        """Set frequency penalty

        Args:
            frequency_penalty (float): _description_
        """
        self.frequency_penalty = frequency_penalty

    def set_presence_penalty(self, presence_penalty: float):
        """Set presence penalty

        Args:
            presence_penalty (float): _description_
        """
        self.presence_penalty = presence_penalty

    def set_stop_token(self, stop_token: str):
        """Set stop token

        Args:
            stop_token (str): _description_
        """
        self.stop_token = stop_token

    def set_top_k(self, top_k: int):
        """Set top k

        Args:
            top_k (int): _description_
        """
        self.top_k = top_k

    def set_top_p(self, top_p: float):
        """Set top p

        Args:
            top_p (float): _description_
        """
        self.top_p = top_p


    def set_seed(self, seed: int):
        """Set seed

        Args:
            seed ([type]): [description]
        """
        self.seed = seed

    def metrics(self) -> str:
        """
        Metrics

        Returns:
            str: _description_
        """
        _sec_to_first_token = self._sec_to_first_token()
        _tokens_per_second = self._tokens_per_second()
        _num_tokens = self._num_tokens(self.history)
        _time_for_generation = self._time_for_generation(self.history)

        return f"""
        SEC TO FIRST TOKEN: {_sec_to_first_token}
        TOKENS/SEC: {_tokens_per_second}
        TOKENS: {_num_tokens}
        Tokens/SEC: {_time_for_generation}
        """

    def time_to_first_token(self, prompt: str) -> float:
        """Time to first token

        Args:
            prompt (str): _description_

        Returns:
            float: _description_
        """
        start_time = time.time()
        self.track_resource_utilization(
            prompt
        )  # assuming `generate` is a method that generates tokens
        first_token_time = time.time()
        return first_token_time - start_time

    def generation_latency(self, prompt: str) -> float:
        """generation latency

        Args:
            prompt (str): _description_

        Returns:
            float: _description_
        """
        start_time = time.time()
        self.run(prompt)
        end_time = time.time()
        return end_time - start_time

    def throughput(self, prompts: List[str]) -> float:
        """throughput

        Args:
            prompts (): _description_

        Returns:
            float: _description_
        """
        start_time = time.time()
        for prompt in prompts:
            self.run(prompt)
        end_time = time.time()
        return len(prompts) / (end_time - start_time)