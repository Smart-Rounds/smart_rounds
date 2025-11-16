from TTS.api import TTS
import os
import re
from core.base_service import BaseService

class Narrator(BaseService):
    def __init__(self):
        super().__init__()
        self.tts = TTS(self.settings.tts_model, gpu=False)
        self.tts.to("cpu")

    def _strip_speaker_labels(self, text: str) -> str:
        """Remove leading 'Speaker:' labels so TTS only reads the spoken content.

        Examples:
        - "Dr. A: This is a line" -> "This is a line"
        - "Dr. Jennifer Hill: Welcome..." -> "Welcome..."
        """
        cleaned_lines = []
        for line in text.splitlines():
            # Strip patterns like "Anything:", e.g. "Dr. A:", "Dr. Smith:"
            cleaned = re.sub(r"^\s*[^:]+:\s*", "", line)
            cleaned_lines.append(cleaned)
        return "\n".join(cleaned_lines)

    def run(self, summary: str, output_path: str) -> str:
        self.logger.info(f"Generating podcast using {self.settings.tts_model}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tts_text = self._strip_speaker_labels(summary)
        self.tts.tts_to_file(text=tts_text, file_path=output_path)
        return output_path
