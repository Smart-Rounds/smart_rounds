import datetime, os
from modules.transcriber import Transcriber
from modules.summarizer import Summarizer
from modules.narrator import Narrator
from core.base_service import BaseService

try:
    from modules.deidentifier import Deidentifier
except ImportError:
    Deidentifier = None


class SmartRoundsPipeline(BaseService):
    def __init__(self, use_deidentifier: bool = True):
        super().__init__()
        self.transcriber = Transcriber()
        self.summarizer = Summarizer()
        self.narrator = Narrator()
        self.use_deidentifier = use_deidentifier

        if use_deidentifier and Deidentifier:
            self.deidentifier = Deidentifier()
        else:
            self.deidentifier = None

    def _timestamp_filename(self, prefix: str, ext: str) -> str:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{now}.{ext}"

    def run(self, audio_path: str):
        self.logger.info(f"Pipeline started for {audio_path}")
        transcript = self.transcriber.run(audio_path)
        if self.deidentifier:
            self.logger.info("Running de-identification step")
            masked = self.deidentifier.run(transcript)
        else:
            self.logger.warning("Skipping de-identification step")
            masked = transcript

        summary = self.summarizer.run(masked)

        output_name = self._timestamp_filename("podcast", "wav")
        output_path = os.path.join(self.settings.output_audio_dir, output_name)
        podcast_path = self.narrator.run(summary, output_path)

        self.logger.info("Pipeline completed successfully")
        return masked, summary, podcast_path
