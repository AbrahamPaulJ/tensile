"""Builds demo/video/tensile-demo.mp4 from what tts.py, record.mjs and music.py wrote: the captured frames
with their real timestamps, the narration placed at the moment each beat started, the music ducked under
it, loudness set for the web, and word-by-word captions burned in by ffmpeg-skill's caption tool, so the
video works with the sound off. Also writes tensile-demo.srt.

usage: python demo/video/assemble.py <path to a checkout of kajisho5/ffmpeg-skill>
       (needs ffmpeg with libass; standard library only here)
"""

import json
import pathlib
import re
import subprocess
import sys
import wave
from array import array

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work"
SKILL = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not SKILL or not (SKILL / "scripts" / "caption.py").exists():
    sys.exit("usage: python demo/video/assemble.py <ffmpeg-skill checkout>")
timeline = json.loads((WORK / "timeline.json").read_text(encoding="utf-8"))
frames, beats, finished = timeline["frames"], timeline["beats"], timeline["finished"]
t0 = frames[0]["t"]
TOTAL = finished - t0

# Video: each frame lasts until the next one was painted.
with open(WORK / "frames.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("ffconcat version 1.0\n")
    for i, frame in enumerate(frames):
        end = frames[i + 1]["t"] if i + 1 < len(frames) else finished
        f.write("file 'frames/%s'\nduration %.4f\n" % (frame["file"], max(end - frame["t"], 0.001)))
    f.write("file 'frames/%s'\n" % frames[-1]["file"])

# Narration: one track, every clip at its beat's start time.
rate = json.loads((WORK / "durations.json").read_text())["rate"]
track = array("h", bytes(2 * int(TOTAL * rate)))
for beat in beats:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(WORK / "audio" / (beat["id"] + ".wav")), "-ac", "1",
                    "-ar", str(rate), "-c:a", "pcm_s16le", str(WORK / "clip.wav")], check=True)
    with wave.open(str(WORK / "clip.wav"), "rb") as w:
        clip = array("h", w.readframes(w.getnframes()))
    at = int((beat["start"] - t0) * rate)
    end = min(at + len(clip), len(track))
    track[at:end] = clip[: end - at]
with wave.open(str(WORK / "narration.wav"), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(track.tobytes())


def stamp(seconds):
    ms = int(round(seconds * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms % 3600000 // 60000, ms % 60000 // 1000, ms % 1000)


def chunks(text, limit=64):
    """Short phrases: whole sentences or clauses, one caption line or two at most."""
    parts = re.split(r"(?<=[.?!:])\s+", text)
    out = []
    for part in parts:
        while len(part) > limit:
            cut = part.rfind(", ", 0, limit)
            cut = cut + 1 if cut > 20 else part.rfind(" ", 0, limit)
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        out.append(part)
    return [p for p in out if p]


# Beats with a full-frame card have no captions: the card already says it.
quiet = {b["id"] for b in json.loads((HERE / "script.json").read_text(encoding="utf-8"))["beats"] if b.get("captions") is False}
srt = []
for beat in beats:
    if beat["id"] in quiet:
        continue
    pieces = chunks(beat["text"])
    weight = sum(len(p) for p in pieces)
    at = beat["start"] - t0
    for piece in pieces:
        length = beat["speech"] * len(piece) / weight
        srt.append("%d\n%s --> %s\n%s\n" % (len(srt) + 1, stamp(at), stamp(at + length), piece))
        at += length
(HERE / "tensile-demo.srt").write_text("\n".join(srt), encoding="utf-8", newline="\n")

# Picture and sound, without captions. The music ducks whenever the voice speaks.
mix = ("[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[voice][key];"
       "[2:a]volume=0.32[bed];"
       "[bed][key]sidechaincompress=threshold=0.015:ratio=10:attack=15:release=450[duck];"
       "[voice][duck]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,"
       "aresample=48000[a]")
subprocess.run([
    "ffmpeg", "-y", "-v", "warning", "-f", "concat", "-safe", "0", "-i", "frames.txt", "-i", "narration.wav",
    "-i", "music.wav", "-filter_complex", mix, "-map", "0:v", "-map", "[a]",
    "-vf", "scale=1920:1080:flags=lanczos,fps=30,format=yuv420p",
    "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-c:a", "aac", "-b:a", "160k", "-shortest", "plain.mp4",
], check=True, cwd=str(WORK))

# Captions: ffmpeg-skill, word by word, the spoken word lit, timed by the voice's own energy.
subprocess.run([
    sys.executable, str(SKILL / "scripts" / "caption.py"), str(WORK / "plain.mp4"),
    "--srt", str(HERE / "tensile-demo.srt"), "--font", "Geist", "--fonts-dir", str(HERE / "fonts"), "--bold",
    "--size", "18", "--max-lines", "2", "--box", "--margin", "14",
    "--karaoke", "--karaoke-style", "word", "--karaoke-timing", "energy", "--highlight-color", "7CC4FF",
    "--animate", "pop", "--preset", "slow", "--quality", "22", "--overwrite", "-o", str(WORK / "captioned.mp4"),
], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(WORK / "captioned.mp4"), "-c", "copy",
                "-movflags", "+faststart", str(HERE / "tensile-demo.mp4")], check=True)
print("%.1f s, %d frames, %.1f MB" % (TOTAL, len(frames), (HERE / "tensile-demo.mp4").stat().st_size / 1e6))
