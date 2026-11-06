# Synthoor

A Toy Software Synth written in Python.

```sh
pip install synthoor
```

### Setup

Hope you have [Python>=3.10](https://www.python.org/downloads/) already, Once you got that you can go ahead and [install uv](https://docs.astral.sh/uv/getting-started/installation/) to get everything set up locally.

```sh
uv sync
uv run jupyter notebook
```

Try to get a local venv setup working because we need the sounddevice module to run the package but if you are not able to get it working, you can also install [Docker](https://www.docker.com/get-started/) and [docker-compose](https://docs.docker.com/compose/install/)

```sh
docker-compose up
```

Or

```
docker build -t synthoor .
docker run -it --rm -p 8888:8888 synthoor
```

**Extra:** The last section of the tutorial directly uses the package which would not directly work with docker because it needs a sounddevice, If you are on a Linux machine with PortAudio you can add `--device /dev/snd` to the docker run command and for MacOs follow this instruction: https://devops.datenkollektiv.de/running-a-docker-soundbox-on-mac.html


### Play it from your keyboard

`examples/keyboard.py` turns your computer keyboard into a three-octave synth, with a live oscilloscope of what you're playing.

```sh
uv sync
uv run python examples/keyboard.py
```

`uv sync` installs [pygame](https://www.pygame.org/), which draws the window. With pip, run `pip install pygame` first.

Hold a key to play a note and let go to release it. Chords work.

| Keys              | Notes   |
| ----------------- | ------- |
| `q w e r t y u i` | C5 – C6 |
| `a s d f g h j k` | C4 – C5 |
| `z x c v b n m ,` | C3 – C4 |

The window has two screens, and keys only reach it while it's focused.

- **Play** shows a live trace of the output. ←/→ change the time span, from 5 ms (single cycles) to 1 s (the envelopes and LFO at work).
- **Sound settings** (press Tab): ↑/↓ choose a control, 1–4 jump to a column, ←/→ adjust it, or click and drag with the mouse. Notes still sound here, so you can hear each change. Esc goes back; Esc on the play screen quits.

What you can shape:

- **Three oscillators**, each with a waveform (sine, triangle, saw, square), level, pitch (±24 semitones), detune (±50 cents), its own ADSR envelope (linear or exponential), and a switch for whether the LFO acts on it.
- **LFO**: waveform, rate (0.1–20 Hz), and how much it moves the pitch (vibrato), the volume (tremolo) and the FM index.
- **FM**: the modulator's frequency as a ratio of the note's, and the index.
- **AM**: ratio and depth.

Envelope changes apply from the next note; everything else changes notes already sounding. Settings reset when you quit.


### Tutorial

This was primarily a learning project and to go deeper I made a Youtube series about building the project: https://www.youtube.com/playlist?list=PLc37Ie8gl3e4 I have also presented offline tutorials at [Pycon US 2025](https://us.pycon.org/2025/schedule/presentation/19/), [EuroPython 2026](https://ep2026.europython.eu/session/build-a-synthesizer-with-python) and [Pycon Spain 2026](https://2026.es.pycon.org/en/programa/#session/QAWGUN/)


### Usage

Initialize a simple Sine wave oscillator and plot it

```python
import sounddevice as sd
from synthoor import Oscillator

osc0 = Oscillator('sine')
osc_play = osc0(
    freq = 200,
    frames = 30000
)
sd.play(osc_play)

plt.plot(osc0(
    freq = 10,
    frames = 30000
))
```


Modulate the sine wave and plot it

```python
osc0 = Oscillator('sine', 100)
osc1 = Oscillator('sine', 1000)
a0 = osc0() * 12
a1 = osc1(a0)
plt.plot(a1)
```

Programming Synthesizers as Classes

```python
class SimpleSynth(GatedSound):

    def __init__(self):
        super().__init__()
        self.osc0 = Oscillator('sine')

    def forward(self):

        g0 = self.gate()
        a0 = self.osc0()

        return a0 * g0

s = SimpleSynth()
s.play(
    duration=0.5
)
```

Sound modules are ripped off [Jupylet](https://github.com/nir/jupylet/)
