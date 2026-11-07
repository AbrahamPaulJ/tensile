# How the video is made

`tensile-demo.mp4` is the built page, driven in Chrome in real time: the digit is drawn with the mouse, every
transaction runs on the EVM in the tab, and nothing on the page is sped up or staged. The dark cards and the
pointer are added by the recorder. The voice is synthetic: Kokoro, run locally.

```
powershell -File demo/video/setup_tts.ps1                                  # once: a venv and the 350 MB voice model, in the user's cache
%LOCALAPPDATA%\tensile-tts\venv\Scripts\python demo/video/tts.py           # script.json to work/audio, about 7 min on a CPU
node demo/video/record.mjs                                                 # the page to work/frames, 3 min, needs Chrome
python demo/video/assemble.py                                              # frames, voice and captions to the mp4; needs ffmpeg with libass
```

To change a word, edit `script.json`: `text` is the caption and `say` what the voice is given, where it needs help.
Each beat in it has an action of the same name in `record.mjs`. To use a real voice, record one wav per beat into
`work/audio/<id>.wav`, set their lengths in `work/durations.json`, and run the last two steps.
