"""Writes work/music.wav: a quiet synthesised bed for the video, made here from sine waves so it has no
licence of its own. A minor pad and a soft arpeggio, a low hit when the claim is slashed, and a fade at
the end. assemble.py ducks it under the narration.

usage: python demo/video/music.py      (after record.mjs; needs NumPy)
"""

import json
import pathlib
import wave

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work"
RATE = 48000
BPM = 84
BEAT = 60 / BPM

timeline = json.loads((WORK / "timeline.json").read_text(encoding="utf-8"))
t0 = timeline["frames"][0]["t"]
total = timeline["finished"] - t0
starts = {b["id"]: b["start"] - t0 for b in timeline["beats"]}
n = int(total * RATE)
rng = np.random.default_rng(7)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def tone(freq, length, harmonics=(1.0, 0.35, 0.18, 0.08), detune=0.0):
    t = np.arange(int(length * RATE)) / RATE
    out = np.zeros_like(t)
    for k, a in enumerate(harmonics, 1):
        for d in (-detune, detune) if detune else (0.0,):
            out += a * np.sin(2 * np.pi * freq * k * (1 + d) * t + rng.uniform(0, 6.28))
    return out


def envelope(length, attack, release):
    m = int(length * RATE)
    env = np.ones(m)
    a, r = min(m, int(attack * RATE)), min(m, int(release * RATE))
    env[:a] = np.linspace(0, 1, a) ** 2
    env[m - r:] *= np.linspace(1, 0, r) ** 2
    return env


def place(track, clip, at):
    i = int(at * RATE)
    if i >= len(track):
        return
    j = min(len(track), i + len(clip))
    track[i:j] += clip[: j - i]


# A minor: Am, F, C, G, one bar each (MIDI notes).
CHORDS = [(57, 60, 64, 69), (53, 57, 60, 65), (48, 55, 60, 64), (55, 59, 62, 67)]
BAR = 4 * BEAT
pad, arp, bass = np.zeros(n), np.zeros(n), np.zeros(n)
bar = 0
while bar * BAR < total:
    chord = CHORDS[bar % len(CHORDS)]
    at = bar * BAR
    length = BAR + 1.2
    for note in chord:
        place(pad, tone(hz(note), length, detune=0.0025) * envelope(length, 0.9, 1.4), at)
    place(bass, tone(hz(chord[0] - 12), length, harmonics=(1.0, 0.2)) * envelope(length, 0.3, 1.0), at)
    # Eighth-note arpeggio an octave up, from the fourth bar, so the opening is calm.
    if bar >= 3:
        order = (0, 1, 2, 3, 2, 1, 2, 3)
        for k, idx in enumerate(order):
            length = 0.9
            t = np.arange(int(length * RATE)) / RATE
            pluck = tone(hz(chord[idx] + 12), length, harmonics=(1.0, 0.25)) * np.exp(-t * 6.5)
            place(arp, pluck * (0.8 if k % 2 else 1.0), at + k * BEAT / 2)
    bar += 1

mix = 0.11 * pad / 4 + 0.05 * arp + 0.09 * bass

# The slash: a falling low hit and a short noise swell, on the beat's first word.
if "slashed" in starts:
    length = 2.2
    t = np.arange(int(length * RATE)) / RATE
    sweep = np.sin(2 * np.pi * (90 * t - 18 * t * t)) * np.exp(-t * 2.4)
    noise = rng.standard_normal(len(t)) * np.exp(-t * 9)
    kernel = np.ones(40) / 40
    place(mix, 0.55 * sweep + 0.06 * np.convolve(noise, kernel, mode="same"), starts["slashed"] - 0.05)

# A little room: convolve with a short decaying noise tail.
tail = rng.standard_normal(int(1.6 * RATE)) * np.exp(-np.arange(int(1.6 * RATE)) / RATE * 3.2)
tail /= np.sqrt((tail ** 2).sum())
size = 1 << int(np.ceil(np.log2(len(mix) + len(tail))))
wet = np.fft.irfft(np.fft.rfft(mix, size) * np.fft.rfft(tail, size), size)[: len(mix)]
left = mix + 0.35 * wet
right = np.roll(mix, int(0.011 * RATE)) + 0.35 * np.roll(wet, int(0.017 * RATE))

fade_in, fade_out = int(1.5 * RATE), int(4.0 * RATE)
gain = np.ones(n)
gain[:fade_in] = np.linspace(0, 1, fade_in)
gain[n - fade_out:] = np.linspace(1, 0, fade_out)
stereo = np.stack([left * gain, right * gain], axis=1)
stereo /= max(1e-9, np.abs(stereo).max()) / 0.7
with wave.open(str(WORK / "music.wav"), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(RATE)
    w.writeframes((stereo * 32767).astype("<i2").tobytes())
print("music %.1f s" % total)
