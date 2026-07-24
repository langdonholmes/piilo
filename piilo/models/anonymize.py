from typing import Optional

from pydantic import BaseModel


class AnonymizeRequest(BaseModel):
    raw_text: str
    entities: Optional[list] = None
    language: Optional[str] = "en"


class AnonymizeResponse(BaseModel):
    anonymized_text: str
