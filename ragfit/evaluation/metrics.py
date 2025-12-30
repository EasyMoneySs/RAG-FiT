import re
import string
import unicodedata
from collections import Counter, defaultdict

import regex

from .base import MetricBase


class HFEvaluate(MetricBase):
    """
    Wrapper class around `evaluate` metrics; easy to use, only need metric names.
    """

    def __init__(self, key_names, metric_names: list[str], **kwargs):
        """
        Args:
            key_names (dict): A dictionary containing the field names.
            metric_names (list[str]): A list of metric names.
        """
        import evaluate

        super().__init__(key_names, **kwargs)
        self.metric_names = metric_names
        self.metric = evaluate.combine(metric_names)
        self.local = True

    def measure(self, example):
        """
        Measure the performance of the model on a given example.

        Args:
            example (dict): The example containing input and target values.

        Returns:
            dict: The performance metric(s) computed for the example.
        """
        input = example[self.field]
        target = example[self.target]

        if isinstance(target, list):
            results = defaultdict(int)
            for tar in target:
                results = {
                    k: max(v, results[k])
                    for k, v in self.metric.compute(
                        predictions=[input], references=[tar]
                    ).items()
                }
            return results
        else:
            return self.metric.compute(predictions=[input], references=[target])


class Classification(MetricBase):
    """
    Metrics for classification answers: accuracy, precision, recall, F1; macro-averaged.

    mapping: dict - mapping of labels to integers.
        Example: {"true": 1, "false": 0, "maybe": 2}
    else_value: int - value to assign to labels not in the mapping.
    """

    def __init__(
        self, key_names: dict, mapping: dict, else_value: int = 2, **kwargs
    ) -> None:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        super().__init__(key_names, **kwargs)
        self.local = False
        self.mapping = mapping
        self.else_value = else_value
        self.precision_recall_fn = precision_recall_fscore_support
        self.accuracy_fn = accuracy_score

    def in_text(self, text):
        if "yes" in text:
            return 1
        if "no" in text:
            return 0
        return 2

    def measure(self, example: dict):
        inputs = example[self.field]
        targets = example[self.target]

        if isinstance(targets[0], list):
            targets = [t[0] for t in targets]

        inputs = [self.in_text(normalize_text(i).strip()) for i in inputs]

        targets = [
            self.mapping.get(normalize_text(t).strip(), self.else_value) for t in targets
        ]

        precision, recall, f1, _ = self.precision_recall_fn(
            targets, inputs, average="macro"
        )
        accuracy = self.accuracy_fn(targets, inputs)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


def normalize_text(s):
    """
    Normalize the given text by lowercasing it, removing punctuation, articles, and extra whitespace.

    Args:
        s (str): The text to be normalized.

    Returns:
        str: The normalized text.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class F1(MetricBase):
    """
    Implementing F1 based on code from Kilt.
    """

    def __init__(self, key_names, **kwargs) -> None:
        """Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        super().__init__(key_names, **kwargs)
        self.local = True

    @staticmethod
    def _f1(prediction, ground_truth):
        prediction_tokens = normalize_text(prediction).split()
        ground_truth_tokens = normalize_text(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def measure(self, example: dict):
        input = example[self.field]
        target = example[self.target]

        assert isinstance(input, str), f"Generated text should be a string: {input}"
        if not isinstance(target, list):
            target = [target]

        scores = [self._f1(input, t) for t in target]
        return {"F1": max(scores)}


class EM(MetricBase):
    """
    Implementing Exact Match based on code from Kilt.
    """

    def __init__(self, key_names, **kwargs) -> None:
        """Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        super().__init__(key_names, **kwargs)
        self.local = True

    def measure(self, example: dict):
        input = example[self.field]
        target = example[self.target]

        assert isinstance(input, str), f"Generated text should be a string: {input}"
        if not isinstance(target, list):
            target = [target]

        scores = [normalize_text(input) == normalize_text(t) for t in target]
        return {"EM": int(max(scores))}


class StringEM(MetricBase):
    """
    Implementing String Exact Match.

    Used in ASQA to evaluate whether the annoated short answers appear in the
    generated answer as sub-strings.
    """

    def __init__(self, key_names: dict, **kwargs) -> None:
        """
        Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        super().__init__(key_names, **kwargs)
        self.local = True

    def measure(self, example: dict):
        input = example[self.field]
        target = example[self.target]

        assert isinstance(input, str), f"Generated text should be a string: {input}"
        assert isinstance(target[0], list), f"Target should be a list of lists: {target}"

        input = normalize_text(input)
        scores = [any(cand in input for cand in item) for item in target]

        return {"StringEM": sum(scores) / len(scores)}


class SimpleTokenizer(object):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


class RecallEM(MetricBase):
    """
    Implementing EM as in XRAG.
    """

    def __init__(self, key_names, **kwargs) -> None:
        """Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        super().__init__(key_names, **kwargs)
        self.local = True

    @staticmethod
    def _normalize(text):
        return unicodedata.normalize("NFD", text)

    def has_answer(self, answers, text, tokenizer=SimpleTokenizer()):
        """Check if a document contains an answer string."""
        text = self._normalize(text)
        text = tokenizer.tokenize(text, uncased=True)

        for answer in answers:
            answer = self._normalize(answer)
            answer = tokenizer.tokenize(answer, uncased=True)
            for i in range(0, len(text) - len(answer) + 1):
                if answer == text[i : i + len(answer)]:
                    return True
        return False

    def measure(self, example: dict):
        input = example[self.field]
        target = example[self.target]

        assert isinstance(input, str), f"Generated text should be a string: {input}"

        if not isinstance(target, list):
            target = [target]

        scores = self.has_answer(target, input)
        return {"recallEM": int(scores)}


class BERTScore(MetricBase):
    """
    BERTScore metric, based on the BERTScore library.
    """

    def __init__(self, key_names: dict, model="microsoft/deberta-large-mnli", **kwargs):
        """Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
            model (str, optional): The name of the BERT model to use. Defaults to "microsoft/deberta-large-mnli".
        """
        super().__init__(key_names, **kwargs)
        from bert_score import BERTScorer

        self.scorer = BERTScorer(model, lang="en", rescale_with_baseline=True)
        self.local = True

    def measure(self, example):
        input = example[self.field]
        target = example[self.target]

        if not isinstance(target, list):
            target = [target]

        scores = [self.scorer.score([input], [t])[2].item() for t in target]

        return {"BERTScore-F1": max(scores)}


class Semantic(MetricBase):
    """
    Semantic similarity between label and answer using a cross-encoder.
    """

    def __init__(
        self,
        key_names: dict,
        model: str = "vectara/hallucination_evaluation_model",
        **kwargs,
    ) -> None:
        """
        Initializes an instance of the class.

        Args:
            key_names (dict): A dictionary containing the field names.
            model (str, optional): The name of the BERT model to use.
        """
        super().__init__(key_names, **kwargs)

        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)
        self.local = True

    def measure(self, example):
        input = example[self.field]
        target = example[self.target]
        if not isinstance(target, list):
            target = [target]

        scores = self.model.predict([[input, t] for t in target])

        return {"Semantic": max(scores)}


class ListF1(MetricBase):
    """
    F1, Precision, and Recall for list/set generation tasks.
    Treats the answer as a set of items and calculates overlap with the ground truth set.
    """

    def __init__(self, key_names, **kwargs) -> None:
        super().__init__(key_names, **kwargs)
        self.local = True

    def _parse_list(self, text):
        """
        Parse a string into a set of normalized items.
        Handles list-like strings "[item1, item2]" or simple comma-separated "item1, item2".
        """
        if isinstance(text, list):
            # Already a list, just normalize strings
            return set(normalize_text(str(item)) for item in text)
        
        text = str(text).strip()
        # Remove brackets if present
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
            
        # Split by comma
        items = [normalize_text(item) for item in text.split(",") if item.strip()]
        return set(items)

    def measure(self, example: dict):
        input_text = example[self.field]  # Model output (already extracted by RegexAnswer)
        target = example[self.target]     # Ground truth

        pred_set = self._parse_list(input_text)
        
        # Handle target. It might be a list of strings ["med1", "med2"] 
        # or a list containing a single string representation of a list ["['med1', 'med2']"]
        # or just a list of valid alternatives (which we treat as a single ground truth set for now based on user description)
        
        # For this specific task (drug list generation), we assume 'target' represents THE correct set of drugs.
        # If target is [["med1", "med2"]], we flatten it or take the first element if it's nested.
        # Adjusting logic to be robust: try to form a single ground truth set.
        
        true_set = set()
        if isinstance(target, list):
            # Check if it's a list of alternatives or the list itself
            # Heuristic: if elements are strings, assume it's the list of drugs.
            # If elements are lists, assume it's alternatives (pick first or union? usually just one GT list exists)
             if len(target) > 0 and isinstance(target[0], list):
                 true_set = self._parse_list(target[0])
             else:
                 # It's likely ["med1", "med2"] directly
                 true_set = self._parse_list(target)
        else:
            true_set = self._parse_list(target)

        # Calculate metrics
        intersection = len(pred_set & true_set)
        len_pred = len(pred_set)
        len_true = len(true_set)

        precision = intersection / len_pred if len_pred > 0 else 0.0
        recall = intersection / len_true if len_true > 0 else 0.0
        
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "List-F1": f1,
            "List-Precision": precision,
            "List-Recall": recall
        }


class Jaccard(MetricBase):
    """
    Jaccard Similarity Coefficient for list/set generation tasks.
    J(A, B) = |A ∩ B| / |A ∪ B|
    """

    def __init__(self, key_names, **kwargs) -> None:
        super().__init__(key_names, **kwargs)
        self.local = True

    def _parse_list(self, text):
        """
        Parse a string into a set of normalized items.
        Robustly handles Python list strings "['a', 'b']" using ast.literal_eval,
        and falls back to simple comma splitting for "a, b".
        """
        import ast
        
        if isinstance(text, list):
            return set(normalize_text(str(item)) for item in text)

        text = str(text).strip()
        
        # 1. Try to parse as a Python literal (list/tuple)
        # This handles cases like "['Item A', 'Item B']" correctly, including internal commas if quoted.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return set(normalize_text(str(item)) for item in parsed)
        except (ValueError, SyntaxError):
            pass
            
        # 2. Fallback: Simple comma separation
        # This handles cases like "Item A, Item B" (no quotes)
        # Remove brackets if strictly wrapping the whole string
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
            
        items = [normalize_text(item) for item in text.split(",") if item.strip()]
        return set(items)

    def measure(self, example: dict):
        input_text = example[self.field]  # Model output
        target = example[self.target]     # Ground truth

        pred_set = self._parse_list(input_text)
        
        true_set = set()
        if isinstance(target, list):
             if len(target) > 0 and isinstance(target[0], list):
                 true_set = self._parse_list(target[0])
             else:
                 true_set = self._parse_list(target)
        else:
            true_set = self._parse_list(target)

        intersection = len(pred_set & true_set)
        union = len(pred_set | true_set)

        jaccard = intersection / union if union > 0 else 0.0

        return {"Jaccard": jaccard}

class FinalScore(MetricBase):
    """
    Final Score as used in ASQA: 0.5 * StringEM + 0.5 * F1
    """

    def __init__(self, key_names: dict, **kwargs) -> None:
        """
        Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        super().__init__(key_names, **kwargs)
        self.jaccard = Jaccard(key_names)
        self.listf1 = ListF1(key_names)
        self.local = True
    
    def measure(self, example: dict):
        jaccard_result = self.jaccard.measure(example)
        listf1_result = self.listf1.measure(example)

        final_score = 0.5 * jaccard_result["Jaccard"] + 0.5 * listf1_result["List-F1"]
        return {"FinalScore": final_score}
