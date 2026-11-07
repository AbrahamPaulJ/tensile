"""Synthesises the narration of demo/video/script.json into work/audio/<id>.wav, and writes
work/durations.json. Uses the local Kokoro model set up by setup_tts.ps1; nothing leaves the machine.

usage: <cache>\\venv\\Scripts\\python demo/video/tts.py
"""

import json
import os
import pathlib

import soundfile
from kokoro_onnx import Kokoro

HERE = pathlib.Path(__file__).resolve().parent
CACHE = pathlib.Path(os.environ["LOCALAPPDATA"]) / "tensile-tts"
OUT = HERE / "work" / "audio"
OUT.mkdir(parents=True, exist_ok=True)

script = json.loads((HERE / "script.json").read_text(encoding="utf-8"))
kokoro = Kokoro(str(CACHE / "kokoro-v1.0.onnx"), str(CACHE / "voices-v1.0.bin"))
durations = {}
for beat in script["beats"]:
    samples, rate = kokoro.create(
        beat.get("say", beat["text"]), voice=script["voice"], speed=script["speed"], lang="en-us")
    soundfile.write(str(OUT / (beat["id"] + ".wav")), samples, rate)
    durations[beat["id"]] = len(samples) / rate
    print("%-10s %5.1f s" % (beat["id"], durations[beat["id"]]))
(HERE / "work" / "durations.json").write_text(json.dumps({"rate": rate, "seconds": durations}, indent=1))
print("total %.1f s" % sum(durations.values()))
