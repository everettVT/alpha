from pydantic import BaseModel
from typing import Callable

def BaseTool(BaseModel):
    
    func
    response_model
    
    def __str__()
        


PyfuncTool(BaseTool):


class PyfuncTool(BaseModel):
    func: Callable
    description: str

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def __async_call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return f"{self.func.__name__}: {self.description}"

# Usage
def add(a, b):
    return a + b

wrapped_add = Tool(func=add, description="This function adds two numbers.")
