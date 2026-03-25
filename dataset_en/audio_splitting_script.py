from pathlib import Path
from pydub import AudioSegment

# ========= CONFIG =========
SOURCE_DIR = Path("data/audio/en")
OUTPUT_DIR = Path("data/audio/en_clips")
CLIP_DURATION_MS = 3000  # 3 seconds
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
# ==========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for audio_file in SOURCE_DIR.iterdir():
    if not audio_file.is_file() or audio_file.suffix.lower() not in ALLOWED_EXTENSIONS:
        continue

    audio = AudioSegment.from_file(audio_file)
    total_length = len(audio)  # in milliseconds

    clip_count = total_length // CLIP_DURATION_MS

    for i in range(clip_count):
        start = i * CLIP_DURATION_MS
        end = start + CLIP_DURATION_MS
        clip = audio[start:end]

        output_name = f"{audio_file.stem}_clip_{i+1:03d}.wav"
        output_path = OUTPUT_DIR / output_name
        clip.export(output_path, format="wav")

    print(f"Processed {audio_file.name}: created {clip_count} clips")

print("Done.")