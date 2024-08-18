import json
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
from guardrails.validator_base import ErrorSpan

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.logger import logger

class Policies(str, Enum):
    NO_VIOLENCE_HATE = "O1"
    NO_SEXUAL_CONTENT = "O2"
    NO_CRIMINAL_PLANNING = "O3"
    NO_GUNS_AND_ILLEGAL_WEAPONS = "O4"
    NO_ILLEGAL_DRUGS = "O5"
    NO_ENOURAGE_SELF_HARM = "O6"

@register_validator(name="guardrails/llamaguard_7b", data_type="string")
class LlamaGuard7B(Validator):
    """
    Classifies model inputs or outputs as "safe" or "unsafe" based on certain policies defined by the LlamaGuard-7B model. 
    
    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/llamaguard_7b`        |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        policies (List[Policies]): List of LlamaGuard7B.Policies enum values to enforce. 
        score_threshold (float): Threshold score for the classification. If the score is above this threshold, the input is considered unsafe.
    """  # noqa

    Policies = Policies

    def __init__(
        self,
        policies: Optional[List[Policies]] = None,
        validation_method: Optional[str] = "full",
        on_fail: Optional[Callable] = None,
    ):

        super().__init__(
            on_fail=on_fail,
            validation_method=validation_method,
        )

        try:
            self._policies = [policy.value for policy in policies] if policies else []
        except AttributeError as e:
            raise ValueError("Invalid policies provided. Please provide a list of LlamaGuard7B.Policies enum values.") from e
        

    def validate(self, value: Any, metadata: Dict = {}) -> ValidationResult:

        if not value:
            raise ValueError("Value cannot be empty.")
        
        (classification, subclass) = self._inference(value)

        is_unsafe = classification == "unsafe"

        if is_unsafe:
            error_span = ErrorSpan(
                start=0,
                end=len(value),
                reason=f"Unsafe content: {value}",
            )

            find_policy_violated = next(
                (policy for policy in self.Policies if policy.value == subclass),
                None
            )
            return FailResult(
                error_message=(
                    f"Prompt contains unsafe content. Classification: {classification}, Violated Policy: {find_policy_violated}"
                ),
                error_spans=[error_span],
            )
        else:
            return PassResult()
    
    
    def _inference_local(self, value: str):
        raise NotImplementedError("Local inference is not supported for LlamaGuard7B validator.")

    def _inference_remote(self, value: str) -> ValidationResult:
        """Remote inference method for this validator."""
        request_body = {
            "policies": self._policies,
            "chat": [
                {
                    "role": "user",
                    "content": value
                }
            ]
        }

        response = self._hub_inference_request(json.dumps(request_body), self.validation_endpoint)
    
        status = response.get("status")
        if status != 200:
            detail = response.get("response",{}).get("detail", "Unknown error")
            raise ValueError(f"Failed to get valid response from Llamaguard-7B model. Status: {status}. Detail: {detail}")

        response_data = response.get("response")

        classification = response_data.get("class") 
        subclass = response_data.get("subclass")

        return (classification, subclass)
