"""Detection, debounce, cooldown, and lazy-backend behavior for the wake word."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from openjarvis.speech.wake_word import WakeWordDetector


class _FakeModel:
    """Return a queued score per frame under the configured phrase key."""

    def __init__(self, scores: list[float], *, phrase: str = "hey_jarvis") -> None:
        self._scores = iter(scores)
        self._phrase = phrase
        self.calls = 0

    def predict(self, frame: bytes) -> dict[str, float]:
        self.calls += 1
        return {self._phrase: next(self._scores)}


def _frames(count: int) -> list[bytes]:
    """Placeholder PCM frames; the fake model ignores their contents."""
    return [b"\x00\x00" * 8 for _ in range(count)]


def _detector(scores: list[float], **kwargs) -> WakeWordDetector:
    model = _FakeModel(scores, phrase=kwargs.pop("phrase", "hey_jarvis"))
    frames = _frames(len(scores))
    return WakeWordDetector(
        model=model,
        frame_source=lambda: iter(frames),
        **kwargs,
    )


def test_fires_when_score_crosses_threshold() -> None:
    detector = _detector([0.1, 0.2, 0.8], threshold=0.5)
    assert detector.wait_for_wake() is True


def test_does_not_fire_below_threshold() -> None:
    detector = _detector([0.1, 0.2, 0.49, 0.3], threshold=0.5)
    assert detector.wait_for_wake() is False


def test_threshold_is_inclusive() -> None:
    detector = _detector([0.5], threshold=0.5)
    assert detector.wait_for_wake() is True


def test_on_partial_receives_every_score() -> None:
    detector = _detector([0.1, 0.4, 0.9], threshold=0.5)
    seen: list[float] = []
    detector.wait_for_wake(on_partial=seen.append)
    assert seen == [0.1, 0.4, 0.9]


def test_debounce_requires_release_before_retrigger() -> None:
    # A lingering utterance (already high when a fresh wait begins) must not
    # fire until the score first drops below threshold and re-arms.
    detector = _detector([0.9, 0.9, 0.2, 0.9], threshold=0.5)
    detector._armed = False  # simulate the tail of a prior detection

    fired_at: list[int] = []

    def _probe(score: float) -> None:
        fired_at.append(score)

    assert detector.wait_for_wake(on_partial=_probe) is True
    # First two 0.9s ignored (not armed), 0.2 re-arms, final 0.9 fires.
    assert fired_at == [0.9, 0.9, 0.2, 0.9]


def test_cooldown_suppresses_immediate_retrigger() -> None:
    clock = {"t": 100.0}
    detector = _detector(
        [0.9, 0.2, 0.9],
        threshold=0.5,
        cooldown_seconds=5.0,
        time_source=lambda: clock["t"],
    )

    # First call fires and stamps the cooldown window.
    assert detector.wait_for_wake() is True

    # A second listen within the cooldown must not fire even on a clean
    # high-score utterance (0.2 re-arms, 0.9 is still inside cooldown).
    clock["t"] = 102.0
    detector2_model = _FakeModel([0.2, 0.9])
    detector._model = detector2_model
    detector._frame_source = lambda: iter(_frames(2))
    assert detector.wait_for_wake() is False

    # Once the cooldown elapses, the same audio fires again.
    clock["t"] = 110.0
    detector._model = _FakeModel([0.2, 0.9])
    detector._frame_source = lambda: iter(_frames(2))
    assert detector.wait_for_wake() is True


def test_keyboard_interrupt_from_source_ends_cleanly() -> None:
    def _interrupting_source():
        raise KeyboardInterrupt
        yield b""  # pragma: no cover - generator marker

    detector = WakeWordDetector(
        model=_FakeModel([]),
        frame_source=_interrupting_source,
    )
    assert detector.wait_for_wake() is False


def test_exhausted_source_returns_false() -> None:
    detector = _detector([0.1, 0.2], threshold=0.5)
    assert detector.wait_for_wake() is False


def test_extract_score_prefers_phrase_key() -> None:
    detector = WakeWordDetector(model=object(), frame_source=lambda: iter([]))
    score = detector._extract_score({"alexa": 0.9, "hey_jarvis": 0.3})
    assert score == 0.3


def test_extract_score_falls_back_to_max() -> None:
    detector = WakeWordDetector(model=object(), frame_source=lambda: iter([]))
    assert detector._extract_score({"other": 0.7, "another": 0.2}) == 0.7
    assert detector._extract_score({}) == 0.0


def test_missing_openwakeword_raises_actionable_error(monkeypatch) -> None:
    # Simulate the extra being absent so the real-backend path is exercised.
    monkeypatch.setitem(sys.modules, "openwakeword", None)
    monkeypatch.setitem(sys.modules, "openwakeword.model", None)

    detector = WakeWordDetector(frame_source=lambda: iter(_frames(1)))
    with pytest.raises(RuntimeError) as excinfo:
        detector.wait_for_wake()

    message = str(excinfo.value)
    assert "openwakeword" in message
    assert "OpenJarvis[wake]" in message


def test_real_model_adapter_converts_bytes_to_int16(monkeypatch) -> None:
    # The lazily-imported openwakeword Model receives an int16 ndarray, while
    # the detector core always deals in bytes frames.
    import numpy as np

    captured: dict[str, object] = {}

    class _RecordingModel:
        def predict(self, array):
            captured["array"] = array
            return {"hey_jarvis": 0.99}

    fake_module = SimpleNamespace(
        model=SimpleNamespace(Model=lambda **kwargs: _RecordingModel()),
        utils=SimpleNamespace(download_models=lambda *a, **k: None),
    )
    monkeypatch.setitem(sys.modules, "openwakeword", fake_module)
    monkeypatch.setitem(sys.modules, "openwakeword.model", fake_module.model)

    frame = np.array([1, -2, 3, -4], dtype=np.int16).tobytes()
    detector = WakeWordDetector(frame_source=lambda: iter([frame]), threshold=0.5)

    assert detector.wait_for_wake() is True
    array = captured["array"]
    assert array.dtype == np.int16
    assert list(array) == [1, -2, 3, -4]
