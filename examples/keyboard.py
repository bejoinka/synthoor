"""Play synthoor from the computer keyboard: three octaves, three oscillators
each with its own envelope, an LFO, FM and AM.

Hold a key to sound its note; let go to release it.

    q  w  e  r  t  y  u  i     C5 D5 E5 F5 G5 A5 B5 C6
    a  s  d  f  g  h  j  k     C4 D4 E4 F4 G4 A4 B4 C5
    z  x  c  v  b  n  m  ,     C3 D3 E3 F3 G3 A3 B3 C4

The play screen shows a live trace of what the speakers are playing. Left and
right change how much time it spans, from 5 ms (single cycles) to 1 s (the
envelopes and the LFO at work).

Tab switches between playing and the sound settings. In settings, up/down
picks a control, 1-4 jump to a column, and left/right changes it; or click and
drag with the mouse. The note keys still sound there, so you can hear each
change. Esc goes back to playing; Esc while playing quits.

Oscillators: each note mixes three, each with its own waveform, level, pitch
(in semitones), detune (in cents) and envelope. Try two saws detuned a few
cents apart, or a second oscillator 12 semitones up that fades in slowly.

Envelopes (tutorial 3): attack, decay, sustain and release shape an
oscillator's volume, on linear or exponential curves. Changes apply from the
next note; everything else applies at once, to notes already sounding.

LFO: a slow oscillator, its rate in Hz whatever the key, that wobbles the
pitch (vibrato), the volume (tremolo) and the FM index (a timbre that sweeps),
each by its own amount. It acts on the oscillators whose LFO switch is on, and
starts afresh with every note.

FM and AM (tutorial 2): each modulator runs at a ratio of the note's frequency,
so a sound keeps its character across the keys. Whole-number ratios sound
harmonic, in-between ones (1.4, 3.5) bell-like or metallic, and very low ones
(around 0.02, a few Hz) turn FM into vibrato and AM into tremolo.

Run with:

    uv run python examples/keyboard.py

Keys go to the window, so keep it focused.
"""

import math
import operator
from dataclasses import dataclass, field

import numpy as np
import pygame
from numpy.lib.stride_tricks import sliding_window_view

from synthoor import Envelope, GatedSound, Oscillator, player
from synthoor.config import FPS

# Keyboard rows, top to bottom, and the octave each one starts on.
ROWS = [("qwertyui", 5), ("asdfghjk", 4), ("zxcvbnm,", 3)]
SCALE = [(0, "C"), (2, "D"), (4, "E"), (5, "F"), (7, "G"), (9, "A"), (11, "B"), (12, "C")]

# Key -> (MIDI note, label). Middle C (C4) is MIDI 60. pygame's key code for a
# letter or punctuation key is its character code, so ord() gives it.
KEYMAP = {
    ord(char): (12 * (octave + 1) + step, f"{name}{octave + step // 12}")
    for chars, octave in ROWS
    for char, (step, name) in zip(chars, SCALE)
}

# synthoor Oscillator shape -> name shown.
WAVES = [("sine", "sine"), ("tri", "triangle"), ("saw", "saw"), ("square", "square")]


@dataclass
class Osc:
    """One oscillator's settings, including the envelope that shapes it."""

    wave: str = "sine"
    level: float = 0.0
    pitch: float = 0.0  # semitones
    detune: float = 0.0  # cents
    lfo: bool = True  # whether the LFO wobbles this oscillator
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.8
    release: float = 0.2
    linear: bool = True

    def envelope(self):
        return Envelope(
            attack=self.attack,
            decay=self.decay,
            sustain=self.sustain,
            release=self.release,
            linear=self.linear,
        )


@dataclass
class Patch:
    """The sound settings shared by every voice."""

    osc1: Osc = field(default_factory=lambda: Osc(level=1.0))
    osc2: Osc = field(default_factory=Osc)
    osc3: Osc = field(default_factory=Osc)
    lfo_wave: str = "sine"
    lfo_rate: float = 5.0  # Hz
    lfo_pitch: float = 0.0  # semitones either way
    lfo_volume: float = 0.0  # how far the volume dips
    lfo_fm: float = 0.0  # added to the FM index at the LFO's peak
    fm_ratio: float = 1.0
    fm_index: float = 0.0
    am_ratio: float = 0.02
    am_depth: float = 0.0


def get_setting(patch, path):
    return operator.attrgetter(path)(patch)


def set_setting(patch, path, value):
    owner, _, name = path.rpartition(".")
    setattr(get_setting(patch, owner) if owner else patch, name, value)


class Voice(GatedSound):
    """One sounding note: three oscillators, each shaped by its own envelope,
    mixed; frequency- and amplitude-modulated by two sines; and wobbled by an
    LFO on the oscillators it is switched on for."""

    def __init__(self, patch):
        super().__init__()
        self.patch = patch
        # Separate attributes, not lists: synthoor only passes the buffer size
        # on to Sound objects held directly as attributes.
        self.osc1 = Oscillator()
        self.osc2 = Oscillator()
        self.osc3 = Oscillator()
        # Envelopes are fixed when the note starts; everything else stays live.
        self.env1 = patch.osc1.envelope()
        self.env2 = patch.osc2.envelope()
        self.env3 = patch.osc3.envelope()
        self.fm = Oscillator("sine")
        self.am = Oscillator("sine")
        self.lfo = Oscillator()

    def forward(self):
        p = self.patch
        gate = self.gate()

        # LFO: runs between -1 and 1 at its own rate, whatever the note.
        self.lfo.shape = p.lfo_wave
        lfo = self.lfo(freq=p.lfo_rate)
        vibrato = 2 ** (p.lfo_pitch * lfo / 12)  # up to lfo_pitch semitones either way
        tremolo = 1 - p.lfo_volume * (1 - lfo) / 2  # between 1 and 1 - lfo_volume

        fm = self.fm(freq=p.fm_ratio * self.freq)

        mix, total = 0, 0
        for osc, env, s in (
            (self.osc1, self.env1, p.osc1),
            (self.osc2, self.env2, p.osc2),
            (self.osc3, self.env3, p.osc3),
        ):
            # Run every envelope, even a silent oscillator's, so it is in the
            # right place if that oscillator is turned up mid-note.
            envelope = env(gate)
            if not s.level:
                continue

            # FM: the modulator swings the oscillator's frequency by the same
            # proportion whatever its pitch, so all three stay in tune together.
            # For the note itself that is up to index x modulator frequency
            # either side. The LFO raises the index by up to lfo_fm.
            index = p.fm_index + (p.lfo_fm * (1 + lfo) / 2 if s.lfo else 0)
            swing = 1 + index * p.fm_ratio * fm
            if s.lfo:
                swing = swing * vibrato

            osc.shape = s.wave
            freq = self.freq * 2 ** ((s.pitch + s.detune / 100) / 12)
            out = osc(freq=freq * swing) * envelope
            if s.lfo:
                out = out * tremolo
            mix = mix + s.level * out
            total += s.level
        # Keep the mix no louder than one oscillator at full level.
        mix = mix / max(total, 1)

        # AM: the modulator swings the volume between 1 and 1 - depth.
        am = self.am(freq=p.am_ratio * self.freq)
        return mix * (1 - p.am_depth * (1 - am) / 2)


def note_on(note, patch):
    # A new voice per press, so a released note can finish its tail while the
    # same key is pressed again. The player drops a voice once it falls silent.
    voice = Voice(patch)
    voice.key = note
    voice.gate.open()
    player.add_sound(voice)
    return voice


#
# Settings controls. Each edits one Patch setting, named by its path (such as
# "osc2.detune"), by mouse (press, drag) or by arrow keys (nudge).
#


@dataclass
class Slider:
    """A horizontal slider. Values are rounded to what `fmt` displays, so what
    you see is what plays.

    `scale` maps the slider's travel to values: "lin" evenly, "sq" with finer
    steps near the bottom (for times), "log" by equal ratios.
    """

    label: str
    path: str
    lo: float
    hi: float
    fmt: str
    template: str
    scale: str
    track: pygame.Rect

    def value_at(self, t):
        if self.scale == "log":
            return self.lo * (self.hi / self.lo) ** t
        if self.scale == "sq":
            t = t * t
        return self.lo + t * (self.hi - self.lo)

    def travel_of(self, value):
        if self.scale == "log":
            return math.log(value / self.lo) / math.log(self.hi / self.lo)
        t = (value - self.lo) / (self.hi - self.lo)
        return math.sqrt(max(t, 0)) if self.scale == "sq" else t

    def set_travel(self, patch, t):
        value = self.value_at(min(max(t, 0), 1))
        # Adding 0.0 turns a rounded -0.0 into 0.0, so it shows as "+0".
        set_setting(patch, self.path, float(format(value, self.fmt)) + 0.0)

    def press(self, patch, pos):
        self.set_travel(patch, (pos[0] - self.track.left) / self.track.width)

    drag = press

    def nudge(self, patch, direction):
        # Step along the slider until the rounded value actually changes.
        old = get_setting(patch, self.path)
        t = self.travel_of(old)
        while 0 <= t <= 1 and get_setting(patch, self.path) == old:
            t += direction * 0.005
            self.set_travel(patch, t)

    def hit(self, pos):
        return self.track.inflate(20, 30).collidepoint(pos)

    def draw(self, screen, font, patch, selected):
        value = get_setting(patch, self.path)
        top = self.track.top - 24
        text(screen, font, self.label, ACCENT if selected else DIM, left=self.track.left, top=top)
        text(screen, font, self.template.format(format(value, self.fmt)), TEXT,
             right=self.track.right, top=top)

        knob = self.track.left + min(max(self.travel_of(value), 0), 1) * self.track.width
        pygame.draw.rect(screen, TRACK, self.track, border_radius=3)
        filled = pygame.Rect(self.track.left, self.track.top, knob - self.track.left, self.track.height)
        pygame.draw.rect(screen, ACCENT, filled, border_radius=3)
        pygame.draw.circle(screen, ACCENT, (knob, self.track.centery), 9)


@dataclass
class Choice:
    """One of a few options: click to step through them, or press left/right."""

    label: str
    path: str
    options: list  # (value, name shown) pairs
    row: pygame.Rect

    def index(self, patch):
        return [value for value, _ in self.options].index(get_setting(patch, self.path))

    def press(self, patch, pos):
        self.nudge(patch, 1)

    def drag(self, patch, pos):
        pass

    def nudge(self, patch, direction):
        i = (self.index(patch) + direction) % len(self.options)
        set_setting(patch, self.path, self.options[i][0])

    def hit(self, pos):
        return self.row.collidepoint(pos)

    def draw(self, screen, font, patch, selected):
        name = self.options[self.index(patch)][1]
        text(screen, font, self.label, ACCENT if selected else DIM, left=self.row.left, top=self.row.top)
        text(screen, font, name, TEXT, right=self.row.right, top=self.row.top)


COLS = (24, 336, 648, 960)  # left edge of each settings column
COL_W = 280
WIDTH, HEIGHT = COLS[-1] + COL_W + 24, 640
ENVELOPE_Y = 254  # where each oscillator's envelope section starts


def _track(col, y):
    return pygame.Rect(COLS[col], y, COL_W, 6)


def _row(col, y):
    return pygame.Rect(COLS[col], y, COL_W, 22)


def _oscillator_controls(n):
    col, path = n - 1, f"osc{n}"
    return [
        Choice("wave", f"{path}.wave", WAVES, _row(col, 52)),
        Slider("level", f"{path}.level", 0, 1, ".2f", "{}", "lin", _track(col, 104)),
        Slider("pitch", f"{path}.pitch", -24, 24, "+.0f", "{} st", "lin", _track(col, 150)),
        Slider("detune", f"{path}.detune", -50, 50, "+.0f", "{} cents", "lin", _track(col, 196)),
        Choice("LFO", f"{path}.lfo", [(True, "on"), (False, "off")], _row(col, 222)),
        Slider("attack", f"{path}.attack", 0, 2, ".2f", "{} s", "sq", _track(col, 316)),
        Slider("decay", f"{path}.decay", 0, 2, ".2f", "{} s", "sq", _track(col, 362)),
        Slider("sustain", f"{path}.sustain", 0, 1, ".2f", "{}", "lin", _track(col, 408)),
        Slider("release", f"{path}.release", 0, 4, ".2f", "{} s", "sq", _track(col, 454)),
        Choice("curve", f"{path}.linear", [(True, "linear"), (False, "exponential")], _row(col, 474)),
    ]


COLUMNS = [
    _oscillator_controls(1),
    _oscillator_controls(2),
    _oscillator_controls(3),
    [
        Choice("wave", "lfo_wave", WAVES, _row(3, 52)),
        Slider("rate", "lfo_rate", 0.1, 20, ".2g", "{} Hz", "log", _track(3, 104)),
        Slider("to pitch", "lfo_pitch", 0, 12, ".2f", "±{} st", "sq", _track(3, 150)),
        Slider("to volume", "lfo_volume", 0, 1, ".2f", "{}", "lin", _track(3, 196)),
        Slider("to FM index", "lfo_fm", 0, 10, ".1f", "+{}", "lin", _track(3, 242)),
        Slider("ratio", "fm_ratio", 0.01, 10, ".2g", "×{}", "log", _track(3, 342)),
        Slider("index", "fm_index", 0, 10, ".1f", "{}", "lin", _track(3, 388)),
        Slider("ratio", "am_ratio", 0.01, 10, ".2g", "×{}", "log", _track(3, 488)),
        Slider("depth", "am_depth", 0, 1, ".2f", "{}", "lin", _track(3, 534)),
    ],
]
# All controls in up/down order, and where each column starts in it (for 1-4).
CONTROLS, COLUMN_STARTS = [], []
for column in COLUMNS:
    COLUMN_STARTS.append(len(CONTROLS))
    CONTROLS.extend(column)

HEADERS = [
    ("Oscillator 1", COLS[0], 16),
    ("Oscillator 2", COLS[1], 16),
    ("Oscillator 3", COLS[2], 16),
    ("LFO", COLS[3], 16),
    ("FM  frequency modulation", COLS[3], 286),
    ("AM  amplitude modulation", COLS[3], 432),
]
SCOPE = pygame.Rect(COLS[0], 56, COLS[-1] + COL_W - COLS[0], 536)

BG = (24, 24, 28)
PANEL = (36, 36, 42)
TEXT = (225, 225, 230)
DIM = (140, 140, 150)
ACCENT = (250, 200, 80)
TRACK = (70, 70, 80)


def recent_output(n):
    """The last n samples sent to the speakers (left channel), padded with
    silence if there aren't that many yet. The player keeps the last second."""
    chunks, total = [], 0
    for buffer in reversed(list(player._al)):
        if total >= n:
            break
        chunks.append(buffer[:, 0])
        total += len(buffer)
    out = np.concatenate(chunks[::-1])[-n:] if chunks else np.zeros(0)
    return np.concatenate([np.zeros(n - len(out)), out])


TIMEBASES_MS = [5, 10, 20, 50, 100, 250, 500, 1000]
# Up to this span the scope draws the wave itself, held still; longer spans
# scroll and draw its outline, to show the envelopes and the LFO.
WAVE_UP_TO_MS = 100
# Longer than one cycle of the lowest key (C3, 7.6 ms).
ENVELOPE_WINDOW_S = 0.01


@dataclass
class Scope:
    """A live trace of the output, like an oscilloscope."""

    timebase: int = 2  # index into TIMEBASES_MS
    scale: float = 0.05  # level at the top of the plot

    def zoom(self, direction):
        self.timebase = min(max(self.timebase + direction, 0), len(TIMEBASES_MS) - 1)

    def trace(self):
        ms = TIMEBASES_MS[self.timebase]
        n = int(ms / 1000 * FPS)
        if ms > WAVE_UP_TO_MS:
            return recent_output(n)
        # Start where the wave last crossed zero going up (leaving a full span
        # after it), so a steady note draws the same picture every frame
        # instead of sliding.
        recent = recent_output(n + FPS // 10)
        before = recent[:-n]
        rising = np.flatnonzero((before[:-1] < 0) & (before[1:] >= 0))
        start = rising[-1] + 1 if len(rising) else len(recent) - n
        return recent[start:start + n]

    def draw(self, screen, font, rect):
        pygame.draw.rect(screen, PANEL, rect, border_radius=6)
        for i in range(1, 10):
            x = rect.left + rect.width * i // 10
            pygame.draw.line(screen, BG, (x, rect.top), (x, rect.bottom))
        pygame.draw.line(screen, TRACK, (rect.left, rect.centery), (rect.right, rect.centery))
        text(screen, font, f"{TIMEBASES_MS[self.timebase]} ms", DIM, right=rect.right - 10, top=rect.top + 8)

        samples = self.trace()
        # Grow at once to fit a louder sound; shrink back slowly.
        self.scale = max(self.scale * 0.995, np.abs(samples).max() * 1.1, 0.05)

        plot = rect.inflate(-20, -40)
        ys = plot.centery - np.clip(samples / self.scale, -1, 1) * plot.height / 2
        if TIMEBASES_MS[self.timebase] <= WAVE_UP_TO_MS:
            xs = plot.left + np.arange(len(ys)) * plot.width / (len(ys) - 1)
            pygame.draw.lines(screen, ACCENT, False, np.column_stack([xs, ys]).tolist(), 2)
            return

        # The outline: the highest and lowest point around each pixel column,
        # looking ENVELOPE_WINDOW_S either way so it spans a full cycle.
        per = len(ys) // plot.width
        columns = ys[: per * plot.width].reshape(plot.width, per)
        k = max(1, round(ENVELOPE_WINDOW_S * FPS / per))
        top = sliding_window_view(np.pad(columns.min(1), k, mode="edge"), 2 * k + 1).min(1)
        bottom = sliding_window_view(np.pad(columns.max(1), k, mode="edge"), 2 * k + 1).max(1)
        xs = plot.left + np.arange(plot.width)
        outline = np.concatenate([np.column_stack([xs, top]), np.column_stack([xs, bottom])[::-1]])
        pygame.draw.polygon(screen, ACCENT, outline.tolist())


def text(screen, font, string, colour, **position):
    surface = font.render(string, True, colour)
    screen.blit(surface, surface.get_rect(**position))


def draw_play(screen, fonts, held, scope):
    big, small = fonts
    notes = "  ".join(KEYMAP[key][1] for key in held)
    text(screen, big, notes, ACCENT, left=SCOPE.left, top=16)
    scope.draw(screen, small, SCOPE)

    text(screen, small,
         "keys  z-,  C3-C4  ·  a-k  C4-C5  ·  q-i  C5-C6  ·  "
         "left/right  time span  ·  Tab  sound settings  ·  Esc  quit",
         DIM, left=SCOPE.left, top=HEIGHT - 28)


def draw_envelope(screen, rect, osc):
    """Sketch an oscillator's envelope: attack, decay, a stretch of sustain,
    then release."""
    pygame.draw.rect(screen, PANEL, rect, border_radius=4)
    plot = rect.inflate(-20, -20)

    s = osc.sustain
    hold = max(0.25 * (osc.attack + osc.decay + osc.release), 0.1)
    segments = [(osc.attack, 0, 1), (osc.decay, 1, s), (hold, s, s), (osc.release, s, 0)]
    total = sum(duration for duration, _, _ in segments)

    # The same curves synthoor.envelope draws: a straight line, or an
    # exponential that covers 99% of the distance.
    def shape(u):
        return u if osc.linear else (1 - 0.01**u) / (1 - 0.01)

    points, start = [], 0.0
    for duration, v0, v1 in segments:
        for i in range(21):
            u = i / 20
            x = plot.left + (start + u * duration) / total * plot.width
            y = plot.bottom - (v0 + (v1 - v0) * shape(u)) * plot.height
            points.append((x, y))
        start += duration
    pygame.draw.lines(screen, ACCENT, False, points, 2)


def draw_config(screen, fonts, patch, selected):
    big, small = fonts
    divider = COLS[3] - 16
    pygame.draw.line(screen, PANEL, (divider, 16), (divider, HEIGHT - 44), 2)
    for title, x, y in HEADERS:
        text(screen, big, title, TEXT, left=x, top=y)

    for n, x in enumerate(COLS[:3], start=1):
        pygame.draw.line(screen, PANEL, (x, ENVELOPE_Y), (x + COL_W, ENVELOPE_Y), 2)
        text(screen, small, "Envelope", TEXT, left=x, top=ENVELOPE_Y + 10)
        draw_envelope(screen, pygame.Rect(x, 506, COL_W, 90), getattr(patch, f"osc{n}"))

    for i, control in enumerate(CONTROLS):
        control.draw(screen, small, patch, i == selected)

    text(screen, small,
         "up/down  choose  ·  1-4  jump to column  ·  left/right  adjust  ·  "
         "or click and drag  ·  Tab  play  ·  notes still sound",
         DIM, left=COLS[0], top=HEIGHT - 28)


def main():
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("synthoor")
    fonts = pygame.font.SysFont(None, 28), pygame.font.SysFont(None, 22)

    # SDL starts in text-input mode, which passes keys through macOS's text
    # system and pops up the accent picker when a key is held. We only need
    # raw key-down/key-up, so switch it off.
    pygame.key.stop_text_input()
    # Repeat held keys, so holding an arrow keeps adjusting. Repeats of a held
    # note key are ignored below.
    pygame.key.set_repeat(300, 40)

    # Open the audio stream now so the first key press isn't delayed by it.
    player._init_worker_thread()

    patch = Patch()
    scope = Scope()
    held = {}  # pygame key -> sounding Voice
    config = False  # showing the sound settings rather than the scope
    selected = 0  # index into CONTROLS
    dragging = None  # control under the mouse button
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    config = not config
                    dragging = None
                elif event.key == pygame.K_ESCAPE:
                    if config:
                        config = False
                    else:
                        running = False
                elif event.key in KEYMAP:
                    if event.key not in held:
                        held[event.key] = note_on(KEYMAP[event.key][0], patch)
                elif config and event.key in (pygame.K_UP, pygame.K_DOWN):
                    step = 1 if event.key == pygame.K_DOWN else -1
                    selected = (selected + step) % len(CONTROLS)
                elif config and pygame.K_1 <= event.key < pygame.K_1 + len(COLUMN_STARTS):
                    selected = COLUMN_STARTS[event.key - pygame.K_1]
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    direction = 1 if event.key == pygame.K_RIGHT else -1
                    if config:
                        CONTROLS[selected].nudge(patch, direction)
                    else:
                        scope.zoom(direction)
            elif event.type == pygame.KEYUP and event.key in held:
                held.pop(event.key).gate.close()
            elif config and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hit = next((i for i, c in enumerate(CONTROLS) if c.hit(event.pos)), None)
                if hit is not None:
                    selected, dragging = hit, CONTROLS[hit]
                    dragging.press(patch, event.pos)
            elif event.type == pygame.MOUSEMOTION and dragging and event.buttons[0]:
                dragging.drag(patch, event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = None
            elif event.type == pygame.WINDOWFOCUSGAINED:
                pygame.key.stop_text_input()
            elif event.type == pygame.WINDOWFOCUSLOST:
                # Key-ups go elsewhere once focus is gone; don't leave notes stuck.
                for voice in held.values():
                    voice.gate.close()
                held.clear()
                dragging = None

        screen.fill(BG)
        if config:
            draw_config(screen, fonts, patch, selected)
        else:
            draw_play(screen, fonts, held, scope)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
