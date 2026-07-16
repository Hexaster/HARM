import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, model_validator
from har import config

load_dotenv()

def get_llm() -> ChatOpenAI:
    """
    Get the LLM instance based on the configuration.
    """
    model_id = config.Config.MODEL_ID
    max_tokens = config.Config.MAX_TOKENS

    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_BASE_URL"]

    return ChatOpenAI(
        model=model_id,
        max_tokens=max_tokens,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )

class LLMResponse(BaseModel):
    """Base model for JSON returned by an LLM.

    LLMs occasionally vary the capitalization of keys even when the prompt
    specifies an exact schema. Normalize those keys before Pydantic validates
    the response while keeping the Python field names as the canonical API.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_key_case(cls, value):
        if not isinstance(value, dict):
            return value

        fields_by_case = {name.casefold(): name for name in cls.model_fields}
        normalized = {}
        for key, item in value.items():
            canonical = fields_by_case.get(str(key).casefold(), key)
            # An exactly named key wins if both variants are present.
            if canonical not in normalized or key == canonical:
                normalized[canonical] = item
        return normalized


def text_from_llm(value):
    """Coerce common LLM representations of a text field to one string."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_json(schema: type[BaseModel], text: str) -> BaseModel:
    """
    Parse the given text into a Pydantic model based on the provided schema.
    """
    decoder = json.JSONDecoder()
    last_error = None

    # raw_decode stops at the end of the first JSON value, so markdown fences
    # or explanatory text after the object do not corrupt an otherwise valid
    # response. Try later objects too if an earlier brace starts non-JSON text.
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return schema.model_validate(value)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last_error = exc

    detail = last_error or "no JSON object found"
    raise ValueError(f"Failed to parse JSON: {detail}")
