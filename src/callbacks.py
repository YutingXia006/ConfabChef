from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, List
from langchain_core.messages import BaseMessage

class StreamlitProgressHandler(BaseCallbackHandler):
    def __init__(self, status):
        self.status = status
        self._call_count = 0
    
    def on_chat_model_start(self, serialized: dict, messages: List[List[BaseMessage]], **kwargs: Any) -> None:
        if serialized is None:
            return
        
        self._call_count += 1
        
        # Basierend auf wievielter LLM Call
        if self._call_count == 1:
            self.status.write("🔍 Analyzing your request...")
        elif self._call_count == 2:
            self.status.write("📅 Creating meal plan draft...")
        elif self._call_count == 3:
            self.status.write("🥗 Applying dietary restrictions...")