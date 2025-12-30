import re


class RegexAnswer:
    """
    Extract answers from the text using regular expressions.

    Pattern is the regular expression used to extract the answer.
    Stopping pattern is a string used to split the answer.

    Example:
    `r = RegexAnswer("<ANSWER>: (.*)", "[,.;]")`
    """

    def __init__(self, capture_pattern=None, stopping_pattern=None):
        self.capture_pattern = capture_pattern
        self.stopping_pattern = stopping_pattern

    def __call__(self, text: str):
        """
        Extract the answer from the text.
        """
        if (capture := self.capture_pattern) and capture != "":
            match = re.search(capture, text, re.MULTILINE | re.DOTALL)
            if match:
                text = match.group(1)

        if (stopping := self.stopping_pattern) and stopping != "":
            text = re.split(stopping, text)[0]

        return text


class JSONListAnswer:
    """
    Extract answers from a JSON list string.
    e.g. '["a", "b"]' -> "a, b"
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, text: str):
        import json
        try:
            # Try to find the JSON list part if there's surrounding text
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                text_list = match.group(0)
                data = json.loads(text_list)
                if isinstance(data, list):
                    return ", ".join([str(x) for x in data])
            
            # Fallback: try loading the whole text
            data = json.loads(text)
            if isinstance(data, list):
                return ", ".join([str(x) for x in data])
                
        except Exception:
            pass
            
        return text
