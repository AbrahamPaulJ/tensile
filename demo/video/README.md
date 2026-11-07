# How the video is made

`tensile-demo.mp4` (2:15) is the built page, driven in Chrome in real time: the digit is drawn with the mouse,
every transaction runs on the EVM in the tab, and nothing on the page is sped up or staged. The recorder adds
presentation only: a pointer, a camera (a CSS zoom of the whole page toward what is being talked about), and
full-frame title and number cards. The figures on the cards are read from `docs/numbers.md` and
`docs/mutation-results.md`, never typed. The voice is synthetic: Kokoro, run locally. The music is generated
from sine waves by `music.py`, so it has no licence of its own. Captions are burned in word by word with the
caption tool of [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (MIT), in Geist (`fonts/`, SIL OFL).

```
powershell -File demo/video/setup_tts.ps1                                  # once: a venv and the 350 MB voice model, in the user's cache
git clone --depth 1 https://github.com/kajisho5/ffmpeg-skill <dir>         # once: the caption tool
powershell -File demo/video/make.ps1 -Skill <dir>                          # everything below, about 6 min
```

`make.ps1` runs, in order: `tts.py` (script.json to `work/audio`; unchanged lines are kept, a new line takes
about 30 s on a CPU), `record.mjs` (the page to `work/frames`, 2:15, needs Chrome), `music.py` (needs NumPy) and
`assemble.py` (frames, voice and ducked music to an mp4 at -16 LUFS, then the captions; needs ffmpeg with libass).

To change a word, edit `script.json`: `text` is the caption and `say` what the voice is given, where it needs
help. `captions: false` marks a beat whose card already shows the words. Each beat has an action of the same
name in `record.mjs`, which starts with the narration; the next beat waits for both. To use a real voice,
record one wav per beat into `work/audio/<id>.wav`, set their lengths in `work/durations.json`, and run the
last three steps.
