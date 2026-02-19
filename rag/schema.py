from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RetrievedChunk:
    text: str
    page_number: str  # keep as str because sometimes it's "NA"
    metadata: Dict[str, Any]
    score: float | None = None
