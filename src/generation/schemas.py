from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CitedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1, description="Evidence IDs such as C1 or C2")


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answerable: bool
    claims: list[CitedClaim]
    missing_information: list[str]

    @model_validator(mode="after")
    def consistent_answer(self):
        if self.answerable and not self.claims:
            raise ValueError("An answerable response must contain cited claims")
        if not self.answerable and self.claims:
            raise ValueError("An abstention must not contain unsupported claims")
        return self


@dataclass
class GenerationResult:
    status: Literal["answered", "abstained", "error"]
    answer: GroundedAnswer
    sources: dict[str, dict] = field(default_factory=dict)
    context_chunk_ids: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def abstention(reason: str) -> GenerationResult:
    return GenerationResult("abstained", GroundedAnswer(
        answerable=False, claims=[], missing_information=[reason],
    ))
