from transformers import pipeline
from core.base_service import BaseService
import math, textwrap

class Summarizer(BaseService):
    def __init__(self):
        super().__init__()
        self.summarizer = pipeline("summarization", model=self.settings.summary_model)
        self.prompt_template = (
            "You are an expert medical communicator writing a script to be read aloud by a text-to-speech system. "
            "Summarize the following clinical transcript into a clear, two-speaker dialogue "
            "between Dr. A and Dr. B, preserving important clinical details but using plain language. "
            "Write the dialogue exactly as it should be spoken: expand abbreviations and acronyms so they sound natural "
            "when read aloud (for example, write 'M.D.' instead of 'MD', and avoid raw URLs). Do not include stage "
            "directions or markup, only the spoken lines.\n\n"
            "Transcript:\n{chunk}"
        )

    def _estimate_target_tokens(self, original_minutes):
        # 10 min output per 60 min input
        return int((original_minutes / 60) * 1500)  # rough token estimate

    def _format_conversation(self, summary_text):
        """Turn a narrative summary into a two-speaker dialogue."""
        lines = textwrap.wrap(summary_text, 250)
        script = []
        for i, chunk in enumerate(lines):
            speaker = "Dr. A" if i % 2 == 0 else "Dr. B"
            script.append(f"{speaker}: {chunk.strip()}")
        return "\n".join(script)

    def run(self, text: str, original_minutes: float = 60) -> str:
        self.logger.info(f"Summarizing into dialogue using {self.settings.summary_model}")
        chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
        prompted_chunks = [
            self.prompt_template.format(chunk=chunk)
            for chunk in chunks
        ]
        summaries = [
            self.summarizer(pc, max_length=400, min_length=100, do_sample=False)[0]["summary_text"]
            for pc in prompted_chunks
        ]
        combined = " ".join(summaries)

        # Truncate/expand to target token length
        target_tokens = self._estimate_target_tokens(original_minutes)
        combined = combined[:target_tokens * 5]  # rough char conversion

        return self._format_conversation(combined)
