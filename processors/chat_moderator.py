import logging
import pathway as pw


class ChatModerator:
    # Define the schema for the moderation endpoint
    class ModerationSchema(pw.Schema):
        prompt: str
        masking: bool = pw.column_definition(default_value=False)

    def __init__(self, moderator: pw.UDF):
        self.logger = logging.getLogger(__name__)
        self.moderator = moderator

    def moderate(self, queries: pw.Table) -> pw.Table:
        """
        Example handler that processes foobar queries.
        Takes input queries and returns modified results.
        """
        toxicity = queries.select(
            prompt=pw.this.prompt,
            masking=pw.this.masking,
            is_toxic=self.moderator(pw.this.prompt),
        )
        result = toxicity.select(
            result=pw.apply(lambda x: {"toxic": x}, pw.this.is_toxic)
        )
        return result
