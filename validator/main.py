import json
from typing import Any, Callable, Dict, List, Optional
from guardrails.validator_base import ErrorSpan

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

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
        policies (List[str]): A list of policies that can be any `LlamaGuard7B.POLICY__*` constants.
        score_threshold (float): Threshold score for the classification. If the score is above this threshold, the input is considered unsafe.
    """  # noqa


    POLICY__NO_VIOLENCE_HATE = "O1"
    POLICY__NO_SEXUAL_CONTENT = "O2"
    POLICY__NO_CRIMINAL_PLANNING = "O3"
    POLICY__NO_GUNS_AND_ILLEGAL_WEAPONS = "O4"
    POLICY__NO_ILLEGAL_DRUGS = "O5"
    POLICY__NO_ENOURAGE_SELF_HARM = "O6"

    def __init__(
        self,
        policies: Optional[List[str]] = None,
        validation_method: Optional[str] = "full",
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):

        super().__init__(
            on_fail=on_fail,
            validation_method=validation_method,
            **kwargs,
        )

        self._policies = policies
        

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

            # iterate over self to find any POLICY__* attributes

            find_policy_violated = next(
                (policy_key for policy_key in [
                    "POLICY__NO_VIOLENCE_HATE",
                    "POLICY__NO_CRIMINAL_PLANNING",
                    "POLICY__NO_GUNS_AND_ILLEGAL_WEAPONS",
                    "POLICY__NO_ILLEGAL_DRUGS",
                    "POLICY__NO_ENOURAGE_SELF_HARM",
                    "POLICY__NO_SEXUAL_CONTENT"
                ] if getattr(self,policy_key) == subclass),
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
