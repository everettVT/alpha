from pydantic import BaseModel, Field, HttpUrl

from datetime import datetime
import uuid
import gzip
import json
from enum import Enum


import pyarrow as pa
import daft
import ray

from base import DomainObject
from models import LLM
from tools import Tool


@ray.remote()
class BaseAgent(CustomModel):
    def __init__(
            self,
            name: Optional[str] = Field(..., description="")
            model: Optional[LLM] = Field(...,
            tools: Optional[Tool] = Field(..., 
            embedding: Optional[]
            reasoning: Optional[]
            memory: Optional[List[BasePipeline]]
            logs: daft.DataType.embedding
            metrics:
            tools: Optional[pa.list_[BaseTool]]
    ):
        self.name =  str
        self.models = BaseLanguageModel
        self. = 
        self.embedding = BaseEmbeddingModel
        self.tools = pa.list_()
        self.memory = self.get_query_pipelines()
    

    def register(self):
        unity.tables('data.agents')

    def process(self):


    def run(self, prompt: str):
        MAX_NUM_PENDING_TASKS = 100
        result_refs = []
        for _ in range(NUM_TASKS):
            if len(result_refs) > MAX_NUM_PENDING_TASKS:
        # update result_refs to only
        # track the remaining tasks.
        ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
        ray.get(ready_refs)

        result_refs.append(self.process.remote())

        ray.get(result_refs)
        response = self.lm.complete(prompt)
        return response








    def _select_tool(self, tools , strategy: ):
        print(self.tools)
        

    def complete(
        model: str,
        messages: Optional[DataType.list() = [],
        # Optional OpenAI params
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop=None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        # openai v1.0+ new params
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        # set api_base, api_version, api_key
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        # Optional liteLLM function params
        **kwargs,
        ):

        completion(
            model = self.model_name


        )


    
    


    

