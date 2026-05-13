from pathlib import Path
from faster_whisper import WhisperModel

ROOT = Path(r"c:/Users/manny/Documents/BUS696/Chapman_FSM_DCF")
VIDEOS_DIR = ROOT / "Videos"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".m4a"}


def iter_videos(videos_dir: Path):
    for path in sorted(videos_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def format_timestamp(total_seconds: float) -> str:
    seconds = int(total_seconds)
    mm, ss = divmod(seconds, 60)
    return f"{mm:02d}:{ss:02d}"


def transcribe_one(model: WhisperModel, video_path: Path):
    stem = video_path.stem
    transcript_dir = VIDEOS_DIR / stem / "transcript"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_dir / f"{stem}.transcript.txt"

    segments, info = model.transcribe(str(video_path), vad_filter=True)

    lines = [f"Language: {info.language} (p={info.language_probability:.3f})", ""]
    for segment in segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        lines.append(f"[{start} - {end}] {segment.text.strip()}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved transcript: {out_path}")
    print(f"Detected language: {info.language} ({info.language_probability:.3f})")
    print(f"Transcript lines: {len(lines) - 2}")


def main():
    videos = list(iter_videos(VIDEOS_DIR))
    if not videos:
        print(f"No videos found in: {VIDEOS_DIR}")
        return

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    for video in videos:
        print(f"Transcribing: {video}")
        transcribe_one(model, video)


if __name__ == "__main__":
    main()
