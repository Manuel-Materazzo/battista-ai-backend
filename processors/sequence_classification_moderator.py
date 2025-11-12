import pathway as pw
from pathway.internals import udfs
from pathway.xpacks.llm._utils import _extract_value


class SequenceClassificationModerator(pw.UDF):

    def __init__(
            self,
            model_name: str,
            tokenizer_model_name: str,
            *,
            cache_strategy: udfs.CacheStrategy | None = None,
            **init_kwargs,
    ) -> None:
        # import is nested, otherwise pathway will import whole library at startup even if not used.
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        super().__init__(cache_strategy=cache_strategy)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def __wrapped__(self, prompt: str, **kwargs) -> bool:
        # import is nested, otherwise pathway will import whole library at startup even if not used.
        import torch
        prompt = _extract_value(prompt)

        # Text classification
        batch = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model(batch)

        # idx 0 for neutral, idx 1 for toxic
        logits = output.logits
        prediction = torch.argmax(logits, dim=1)
        return (prediction == 1).item()  # 'Toxic' if prediction == 1 else 'Neutral'

    def __call__(self, prompt: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """Evaluates a prompt for moderation issues using a sequence classification model.
        Args:
            prompt: The input text to be evaluated for moderation.
            **kwargs: Additional keyword arguments passed to the UDF.

        Returns:
            A boolean column indicating whether the prompt was classified as toxic (True) or neutral (False).
        """
        return super().__call__(prompt, **kwargs)
