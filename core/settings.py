from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    environment: str = Field("development", env="ENVIRONMENT")
    whisper_model: str = Field("small", env="WHISPER_MODEL")
    summary_model: str = Field("facebook/bart-large-cnn", env="SUMMARY_MODEL")
    tts_model: str = Field("tts_models/en/ljspeech/tacotron2-DDC", env="TTS_MODEL")
    input_audio_dir: Path = Field(Path("data/input_audio"), env="INPUT_AUDIO_DIR")
    output_audio_dir: Path = Field(Path("data/output_podcasts"), env="OUTPUT_AUDIO_DIR")
    use_gpu: bool = Field(False, env="USE_GPU")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
