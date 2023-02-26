from pydantic import BaseModel
from typing import Optional

class AnonymizeRequest(BaseModel):
    raw_text: str
    entities: Optional[list] = None
    language: Optional[str] = 'en'
    
class AnonymizeResponse(BaseModel):
    anonymized_text: str