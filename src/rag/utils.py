"""
utils.py
Utility function for extracting the answer from a model's text response using regex.
"""
## Utility functions for text processing

import re

def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer:\s*(.*)"
                       ) -> str:
        
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return "Answer not found."