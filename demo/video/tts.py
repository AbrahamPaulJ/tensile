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
    said = beat.get("say", beat["text"])
    # A beat whose words, voice and speed are unchanged keeps its clip.
    key = json.dumps([said, script["voice"], script["speed"]])
    wav, stamp = OUT / (beat["id"] + ".wav"), OUT / (beat["id"] + ".key")
    if wav.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == key:
        info = soundfile.info(str(wav))
        rate, durations[beat["id"]] = info.samplerate, info.frames / info.samplerate
        print("%-10s %5.1f s (kept)" % (beat["id"], durations[beat["id"]]))
        continue
    samples, rate = kokoro.create(said, voice=script["voice"], speed=script["speed"], lang="en-us")
    soundfile.write(str(wav), samples, rate)
    stamp.write_text(key, encoding="utf-8")
    durations[beat["id"]] = len(samples) / rate
    print("%-10s %5.1f s" % (beat["id"], durations[beat["id"]]))
(HERE / "work" / "durations.json").write_text(json.dumps({"rate": rate, "seconds": durations}, indent=1))
print("total %.1f s" % sum(durations.values()))
