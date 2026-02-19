from typing import Dict, Any, List

SESSIONS: Dict[str, Dict[str, Any]] = {}

# per-session conversation history: { session_id: [ {"role": "user"|"assistant", "content": "..."}, ... ] }
HISTORIES: Dict[str, List[Dict[str, str]]] = {}
