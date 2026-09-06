"""Offline "Hey Jarvis" wake-word detection.

Wraps `openwakeword <https://github.com/dscripka/openWakeWord>`_, which ships a
pretrained ``hey_jarvis`` model that runs fully offline. The detector is built
around two injectable seams — a *frame source* and a *model* — so unit tests can
exercise the full detect/cooldown logic with no audio hardware and no model
download. The real backend (microphone + openwakeword) is imported lazily and
only when neither seam is supplied.

Audio conventions mirror :mod:`openjarvis.speech.voice_io`: 16 kHz, mono,
16-bit PCM. openwakeword expects ~80 ms frames (1280 samples at 16 kHz), so the
default chunk is 1280 rather than voice_io's 1024.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

_SAMPLE_RATE = 16000
_CHANNELS = 1
_CHUNK = 1280  # 80 ms at 16 kHz — openwakeword's recommended frame size
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_PHRASE = "hey_jarvis"
_COOLDOWN_SECONDS = 2.0  # ignore re-triggers for this long after a detection


FrameSource = Callable[[], Iterable[bytes]]


class WakeWordModel:
    """Minimal protocol the detector expects from an injected model.

    Implementations return a mapping of ``{model_name: score}`` for one audio
    frame. Both openwakeword and test fakes satisfy this shape.
    """

    def predict(self, frame: bytes) -> Mapping[str, float]:  # pragma: no cover
        raise NotImplementedError


class WakeWordDetector:
    """Detect a spoken wake phrase from a stream of PCM audio frames.

    Parameters
    ----------
    phrase:
        openwakeword model key to listen for. Defaults to ``"hey_jarvis"``.
    threshold:
        Score in ``[0, 1]`` at (or above) which the phrase counts as detected.
    frame_source:
        Optional callable returning an iterable of 16-bit PCM frames (bytes).
        When omitted, a microphone stream is opened lazily via sounddevice.
    model:
        Optional object exposing ``predict(frame) -> {name: score}``. When
        omitted, the pretrained openwakeword model is loaded lazily.
    time_source:
        Monotonic clock used for cooldown/debounce; injectable for tests.
    """

    def __init__(
        self,
        *,
        phrase: str = _DEFAULT_PHRASE,
        threshold: float = _DEFAULT_THRESHOLD,
        sample_rate: int = _SAMPLE_RATE,
        chunk: int = _CHUNK,
        cooldown_seconds: float = _COOLDOWN_SECONDS,
        frame_source: Optional[FrameSource] = None,
        model: Optional[Any] = None,
        time_source: Optional[Callable[[], float]] = None,
    ) -> None:
        self._phrase = phrase
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._chunk = chunk
        self._cooldown_seconds = cooldown_seconds
        self._frame_source = frame_source
        self._model = model
        if time_source is None:
            import time

            time_source = time.monotonic
        self._now = time_source

        # Debounce state. ``_armed`` gates re-triggering: after a detection the
        # detector must observe a below-threshold frame (the utterance ending)
        # before it will fire again, so one "Hey Jarvis" wakes exactly once even
        # though its audio spans many frames and lingers across listen cycles.
        self._armed = True
        self._last_trigger = float("-inf")

    def wait_for_wake(
        self,
        *,
        on_partial: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Block until the wake phrase is detected.

        Reads frames from the (real or injected) source, scores each with the
        model, and returns ``True`` on the first frame that crosses the
        threshold while armed and outside the cooldown window. Returns ``False``
        if the source is exhausted or the wait is interrupted with Ctrl-C, so
        callers can shut a listen loop down cleanly.
        """
        model = self._ensure_model()
        try:
            for frame in self._iter_frames():
                score = self._extract_score(model.predict(frame))
                if on_partial is not None:
                    on_partial(score)
                if score < self._threshold:
                    # Utterance ended (or never began): re-arm for next wake.
                    self._armed = True
                    continue
                if not self._armed:
                    continue
                if self._now() - self._last_trigger < self._cooldown_seconds:
                    continue
                self._armed = False
                self._last_trigger = self._now()
                return True
        except KeyboardInterrupt:
            return False
        return False

    def _extract_score(self, prediction: Mapping[str, float]) -> float:
        """Pull the phrase's score out of a ``{name: score}`` prediction."""
        if not prediction:
            return 0.0
        for name, value in prediction.items():
            if self._phrase in name:
                return float(value)
        return float(max(prediction.values()))

    def _iter_frames(self) -> Iterator[bytes]:
        source = self._frame_source
        if source is None:
            source = self._default_frame_source
        return iter(source())

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self._load_openwakeword_model()
        return self._model

    def _load_openwakeword_model(self) -> Any:
        """Load the pretrained openwakeword model, importing it lazily."""
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "openwakeword is required for the 'Hey Jarvis' wake word. "
                "Install the wake dependencies with: "
                "pip install 'OpenJarvis[wake]'"
            ) from exc

        def _build() -> Any:
            return Model(wakeword_models=[self._phrase], inference_framework="onnx")

        try:
            oww = _build()
        except Exception:
            # Pretrained ONNX weights download on first use; fetch then retry.
            try:
                openwakeword.utils.download_models([self._phrase])
            except Exception:
                openwakeword.utils.download_models()
            oww = _build()
        return _OpenWakeWordModel(oww)

    def _default_frame_source(self) -> Iterator[bytes]:
        """Yield microphone frames via sounddevice (imported lazily)."""
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for wake-word listening. "
                "Install the voice dependencies with: "
                "pip install 'OpenJarvis[speech]'"
            ) from exc

        with sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=_CHANNELS,
            dtype="int16",
            blocksize=self._chunk,
        ) as stream:
            while True:
                raw, _ = stream.read(self._chunk)
                yield bytes(raw)


class _OpenWakeWordModel:
    """Adapt openwakeword's ndarray API to the detector's bytes-frame seam."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, frame: bytes) -> Mapping[str, float]:
        import numpy as np

        array = np.frombuffer(frame, dtype=np.int16)
        return self._model.predict(array)


__all__ = ["WakeWordDetector", "WakeWordModel"]
