import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from har import config

load_dotenv()

def get_llm() -> ChatOpenAI:
    """
    Get the LLM instance based on the configuration.
    """
    model_id = config.Config.model_id
    max_tokens = config.Config.max_tokens

    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_BASE_URL"]

    return ChatOpenAI(
        model=model_id,
        max_tokens=max_tokens,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )

def parse_json(schema: type[BaseModel], text: str) -> BaseModel:
    """
    Parse the given text into a Pydantic model based on the provided schema.
    """
    try:
        # The reflect prompts already instruct a JSON dict output; extract and validate it.
        s, e = text.find("{"), text.rfind("}")
        return schema.model_validate_json(text[s:e + 1])
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}")