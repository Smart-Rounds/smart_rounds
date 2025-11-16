"""
Run the SmartRounds pipeline end-to-end on an audio file.

Usage:
    python run.py <filename>

Example:
    python run.py rounds_recording.wav
"""

import sys
import os
from modules.pipeline import SmartRoundsPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    input_dir = "data/input_audio"
    audio_path = os.path.join(input_dir, filename)

    if not os.path.exists(audio_path):
        print(f"File not found in {input_dir}: {filename}")
        sys.exit(1)

    print(f"Running SmartRounds pipeline on: {audio_path}\n")

    pipeline = SmartRoundsPipeline()
    masked_text, summary_script, podcast_path = pipeline.run(audio_path)

    print("=== CLEANED TRANSCRIPT (first 400 chars) ===")
    print(masked_text[:400].strip() + "...\n")

    print("=== GENERATED TWO-SPEAKER SCRIPT (first 600 chars) ===")
    print(summary_script[:600].strip() + "...\n")

    print(f"✅ Podcast generated at: {podcast_path}")
    print("✨ Pipeline completed successfully.\n")

if __name__ == "__main__":
    main()
