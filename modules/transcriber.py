import whisper
from core.base_service import BaseService

class Transcriber(BaseService):
    def __init__(self):
        super().__init__()
        self.model = whisper.load_model(self.settings.whisper_model)

    def run(self, audio_path: str) -> str:
        self.logger.info(f"Transcribing using model {self.settings.whisper_model}")
        result = self.model.transcribe(audio_path)
        return result["text"]
