"""Tests for double-clap wake detection."""

from __future__ import annotations

import struct
from typing import Iterator

from openjarvis.speech.clap_detect import ClapDetector, _rms


def _pcm(rms_level: int, samples: int = 1024) -> bytes:
    """Build a mono int16 PCM frame with approximately the given RMS."""
    amp = max(-32767, min(32767, int(rms_level)))
    return struct.pack(f"{samples}h", *([amp] * samples))


def test_rms_zero_on_empty():
    assert _rms(b"") == 0.0


def test_rms_nonzero_on_loud_frame():
    assert _rms(_pcm(3000)) > 2000


def test_double_clap_fires():
    clock = {"t": 0.0}

    class Ticking:
        def __iter__(self) -> Iterator[bytes]:
            clock["t"] = 0.0
            yield _pcm(100)
            yield _pcm(4000)
            yield _pcm(100)
            clock["t"] = 0.35
            yield _pcm(4000)
            yield _pcm(100)

    det = ClapDetector(
        threshold=2500,
        gap_min=0.1,
        gap_max=0.9,
        cooldown=0.0,
        frame_source=Ticking(),
        clock=lambda: clock["t"],
    )
    assert det.wait_for_clap() is True


def test_single_clap_does_not_fire():
    clock = {"t": 0.0}

    class Single:
        def __iter__(self) -> Iterator[bytes]:
            clock["t"] = 0.0
            yield _pcm(100)
            yield _pcm(4000)
            yield _pcm(100)
            clock["t"] = 2.0
            yield _pcm(100)
            yield _pcm(100)

    det = ClapDetector(
        threshold=2500,
        gap_min=0.1,
        gap_max=0.9,
        cooldown=0.0,
        frame_source=Single(),
        clock=lambda: clock["t"],
    )
    assert det.wait_for_clap() is False


def test_gap_too_wide_resets():
    clock = {"t": 0.0}

    class Wide:
        def __iter__(self) -> Iterator[bytes]:
            clock["t"] = 0.0
            yield _pcm(4000)
            yield _pcm(100)
            clock["t"] = 1.5
            yield _pcm(4000)
            yield _pcm(100)
            clock["t"] = 3.0
            yield _pcm(100)

    det = ClapDetector(
        threshold=2500,
        gap_min=0.1,
        gap_max=0.9,
        cooldown=0.0,
        frame_source=Wide(),
        clock=lambda: clock["t"],
    )
    assert det.wait_for_clap() is False


def test_keyboard_interrupt_returns_false():
    class Boom:
        def __iter__(self) -> Iterator[bytes]:
            raise KeyboardInterrupt

    det = ClapDetector(frame_source=Boom())
    assert det.wait_for_clap() is False
