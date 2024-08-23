from base import DomainObject
from typing import List, Optional, Any


class Prompt(DomainObject):
    def __init__(self,
        name: Optional[str] = None,
        contents: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop_token: Optional[str] = None,
        *args,
        **kwargs
            ):
        super().__init__(, *args, **kwargs)
        self.contents = contents
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.stop_token = stop_token


