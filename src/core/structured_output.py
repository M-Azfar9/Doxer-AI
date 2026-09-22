"""
Standardized StructuredOutputNode with Repair Loop.
Handles Pydantic schema validation, error correction feedback loop,
and graceful degradation.
"""

from typing import Type, TypeVar, Optional, Any, List
from pydantic import BaseModel, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

T = TypeVar("T", bound=BaseModel)


class StructuredOutputNode:
    """
    Standardized wrapper around .with_structured_output() featuring an automatic
    repair loop when LLM responses violate the Pydantic schema.
    """

    def __init__(
        self,
        llm: Any,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        max_repairs: int = 2
    ):
        self.llm = llm
        self.schema = schema
        self.system_prompt = system_prompt
        self.max_repairs = max_repairs

    def invoke(self, user_prompt: str, context: Optional[str] = None) -> T:
        """Invokes the structured runner with repair-loop validation."""
        messages: List[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        user_content = user_prompt
        if context:
            user_content = f"{user_prompt}\n\nContext:\n{context}"
        messages.append(HumanMessage(content=user_content))

        attempts = 0
        structured_runner = self.llm.with_structured_output(self.schema)

        while attempts <= self.max_repairs:
            try:
                result = structured_runner.invoke(messages)
                if isinstance(result, self.schema):
                    return result
                if isinstance(result, dict):
                    return self.schema(**result)
                # If wrapped or unknown
                return self.schema.model_validate(result)
            except (ValidationError, Exception) as exc:
                attempts += 1
                if attempts > self.max_repairs:
                    print(f"⚠️ [StructuredOutputNode] Exhausted {self.max_repairs} repair attempts: {exc}")
                    raise exc
                
                error_feedback = (
                    f"Your previous response failed validation with error:\n{str(exc)}\n"
                    f"Please correct your output and strictly satisfy the required JSON schema for {self.schema.__name__}."
                )
                print(f"🔧 [StructuredOutputNode] Triggering repair loop (attempt {attempts}/{self.max_repairs})...")
                messages.append(HumanMessage(content=error_feedback))
        
        raise RuntimeError(f"Failed to produce valid structured output for {self.schema.__name__}.")
