"""Double-clap wake detector (offline, no ML).

Listens on the default microphone for two short high-energy bursts within a
short window — the classic "clap clap" pattern. Injectable frame source keeps
unit tests free of hardware.
"""

from __future__ import annotations

import time
from typing import Callable, Iterator, Optional, Protocol

_SAMPLE_RATE = 16000
_CHUNK = 1024  # ~64 ms
_DEFAULT_THRESHOLD = 2500  # RMS of int16 PCM — clap is sharp/loud
_DEFAULT_GAP_MIN = 0.12  # seconds between the two claps (min)
_DEFAULT_GAP_MAX = 0.90  # seconds between the two claps (max)
_DEFAULT_COOLDOWN = 1.5  # seconds after a successful double-clap


class FrameSource(Protocol):
    """Yields raw 16-bit mono PCM bytes."""

    def __iter__(self) -> Iterator[bytes]: ...


def _rms(data: bytes) -> float:
    import struct

    n = len(data) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"{n}h", data[: n * 2])
    return (sum(s * s for s in shorts) / n) ** 0.5


class ClapDetector:
    """Detect a double clap from a stream of PCM audio frames."""

    def __init__(
        self,
        *,
        threshold: float = _DEFAULT_THRESHOLD,
        gap_min: float = _DEFAULT_GAP_MIN,
        gap_max: float = _DEFAULT_GAP_MAX,
        cooldown: float = _DEFAULT_COOLDOWN,
        frame_source: Optional[FrameSource] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._threshold = threshold
        self._gap_min = gap_min
        self._gap_max = gap_max
        self._cooldown = cooldown
        self._frame_source = frame_source
        self._clock = clock or time.monotonic
        self._last_fire = -1e9

    def wait_for_clap(self) -> bool:
        """Block until a double clap is detected. Returns False on KeyboardInterrupt."""
        try:
            first_clap_at: Optional[float] = None
            below = True  # require energy to drop between claps
            for frame in self._iter_frames():
                now = self._clock()
                if now - self._last_fire < self._cooldown:
                    first_clap_at = None
                    below = True
                    continue
                energy = _rms(frame)
                if energy >= self._threshold:
                    if not below:
                        continue  # still in the same loud burst
                    below = False
                    if first_clap_at is None:
                        first_clap_at = now
                    else:
                        gap = now - first_clap_at
                        if self._gap_min <= gap <= self._gap_max:
                            self._last_fire = now
                            return True
                        # Too slow / too fast — treat this as a new first clap
                        first_clap_at = now
                else:
                    below = True
                    if first_clap_at is not None and (
                        now - first_clap_at > self._gap_max
                    ):
                        first_clap_at = None
        except KeyboardInterrupt:
            return False
        return False

    def _iter_frames(self) -> Iterator[bytes]:
        if self._frame_source is not None:
            yield from self._frame_source
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for clap detection. "
                "Install with: pip install 'OpenJarvis[speech]'"
            ) from exc

        with sd.RawInputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=_CHUNK,
        ) as stream:
            while True:
                data, _overflowed = stream.read(_CHUNK)
                yield bytes(data)


__all__ = ["ClapDetector", "FrameSource", "_rms"]
