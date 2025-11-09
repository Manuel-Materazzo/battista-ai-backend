import logging
import pathway as pw
from abc import ABC, abstractmethod


class ChatModerator(ABC):

    # Define the schema for the moderation endpoint
    class ModerationSchema(pw.Schema):
        prompt: str
        masking: bool = pw.column_definition(default_value=False)

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def moderate(self, queries: pw.Table) -> pw.Table:
        pass
