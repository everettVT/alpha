from pydantic import BaseModel
import instructor
import functools

# Model Providers
import vertexai.generative_models as vertexai_models
import vertexai
from openai import OpenAI, AsyncOpenAI
import anthropic
import transformers


import vllm





vertexai.init()


# Vertex AI 
vertexai.generative_models.GenerativeModel("gemini-1.5-pro-preview-0409")
vertexai_client = instructor.from_vertexai(client)
openai_client = instructor.from_openai(openai.OpenAI())

async_openai_client = instructor.from_openai(AsyncOpenAI())


client = instructor.from_openai(
    client=OpenAI(),
    mode=instructor.Mode.TOOLS,
)

await client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        stream=True,
        messages=[
            {"role": "user", "content": f"Extract: `{data.query}`"},
        ],
    )



import openai
import instructor
from typing import Iterable
from pydantic import BaseModel, Field, ConfigDict

client = instructor.from_openai(openai.OpenAI())


class SyntheticQA(BaseModel):
    question: str
    answer: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "What is the capital of France?", "answer": "Paris"},
                {
                    "question": "What is the largest planet in our solar system?",
                    "answer": "Jupiter",
                },
                {
                    "question": "Who wrote 'To Kill a Mockingbird'?",
                    "answer": "Harper Lee",
                },
                {
                    "question": "What element does 'O' represent on the periodic table?",
                    "answer": "Oxygen",
                },
            ]
        }
    )


def get_synthetic_data() -> Iterable[SyntheticQA]:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate synthetic examples"},
            {
                "role": "user",
                "content": "Generate the exact examples you see in the examples of this prompt. ",
            },
        ],
        response_model=Iterable[SyntheticQA],
    ) 

class BaseLanguageModel(BaseModel):
    """Base Text Language Model that defines the interface for all language models"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.stop_token = stop_token
        
        
        # Attributes
        self.history = ""
        self.start_time = None
        self.end_time = None
        self.history = []
        self.memory = {
            "input": [],
            "output": [],
            "task": [],
            "time": [],
            "role": [],
            "model": [],
        }
    
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