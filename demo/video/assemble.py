"""Builds demo/video/tensile-demo.mp4 from what record.mjs and tts.py wrote: the captured frames with
their real timestamps, the narration placed at the moment each beat started, and burned-in captions in a
strip under the page, so the video works with the sound off. Also writes tensile-demo.srt.

usage: python demo/video/assemble.py      (needs ffmpeg with libass; standard library only)
"""

import json
import pathlib
import re
import subprocess
import wave
from array import array

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work"
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

# Audio: one track, every clip at its beat's start time.
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


def chunks(text, limit=118):
    """Whole sentences or clauses, each short enough for two caption lines at most."""
    parts = re.split(r"(?<=[.?!:])\s+", text)
    out, current = [], ""
    for part in parts:
        while len(part) > limit:
            cut = part.rfind(", ", 0, limit)
            cut = cut + 1 if cut > 30 else part.rfind(" ", 0, limit)
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        if current and len(current) + 1 + len(part) > limit:
            out.append(current)
            current = part
        else:
            current = (current + " " + part).strip()
    if current:
        out.append(current)
    return out


srt = []
index = 1
for beat in beats:
    start = beat["start"] - t0
    pieces = chunks(beat["text"])
    weight = sum(len(p) for p in pieces)
    at = start
    for piece in pieces:
        length = beat["speech"] * len(piece) / weight
        srt.append("%d\n%s --> %s\n%s\n" % (index, stamp(at), stamp(at + length), piece))
        index += 1
        at += length
(HERE / "tensile-demo.srt").write_text("\n".join(srt), encoding="utf-8", newline="\n")
(WORK / "captions.srt").write_text("\n".join(srt), encoding="utf-8", newline="\n")

style = "FontName=Segoe UI,FontSize=10.5,PrimaryColour=&H00E8EFEF,BorderStyle=1,Outline=0,Shadow=0,Alignment=2,MarginV=7,MarginL=30,MarginR=30"
subprocess.run([
    "ffmpeg", "-y", "-v", "warning", "-f", "concat", "-safe", "0", "-i", "frames.txt", "-i", "narration.wav",
    "-vf", "scale=1920:960,pad=1920:1080:0:0:color=0x0c1117,fps=30,subtitles=captions.srt:force_style='%s'" % style,
    "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(HERE / "tensile-demo.mp4"),
], check=True, cwd=str(WORK))
print("%.1f s, %d frames, %.1f MB" % (TOTAL, len(frames), (HERE / "tensile-demo.mp4").stat().st_size / 1e6))
