"""``jarvis chat --wake`` wake-word loop integration via CliRunner.

Every audio/model boundary is mocked: the wake detector, STT (record_voice),
TTS (speak), and the engine. No microphone, model download, or network is
touched.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openjarvis.cli._voice_chat import VOICE_EXIT
from openjarvis.cli.chat_cmd import chat
from openjarvis.core.config import JarvisConfig


class _FakeDetector:
    """Return queued wait_for_wake() results; ``False`` once exhausted."""

    def __init__(self, results: list[bool]) -> None:
        self._results = iter(results)
        self.waits = 0
        self.partials: list = []

    def wait_for_wake(self, *, on_partial=None) -> bool:
        self.waits += 1
        try:
            return next(self._results)
        except StopIteration:
            return False


def _enter_base_patches(stack: ExitStack, engine, config, detector) -> None:
    for cm in (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch("openjarvis.speech.wake_word.WakeWordDetector", return_value=detector),
    ):
        stack.enter_context(cm)


def _engine_and_config():
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {"content": "hi there"}
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    return engine, config


class TestWakeFlag:
    def test_help_lists_wake_option(self) -> None:
        result = CliRunner().invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "--wake" in result.output
        assert "--hey-jarvis" in result.output

    def test_wake_implies_voice_banner(self) -> None:
        engine, config = _engine_and_config()
        detector = _FakeDetector([False])  # exit before any turn
        with ExitStack() as stack:
            _enter_base_patches(stack, engine, config, detector)
            record = stack.enter_context(patch("openjarvis.cli.chat_cmd.record_voice"))
            result = CliRunner().invoke(chat, ["--wake", "--model", "test-model"])

        assert result.exit_code == 0
        assert "Wake mode ON" in result.output
        record.assert_not_called()
        assert "Goodbye!" in result.output


class TestWakeCycle:
    def test_full_cycle_then_returns_to_waiting(self) -> None:
        engine, config = _engine_and_config()
        # Wake once (run a turn), then a False wait ends the loop cleanly.
        detector = _FakeDetector([True, False])

        with ExitStack() as stack:
            _enter_base_patches(stack, engine, config, detector)
            record = stack.enter_context(
                patch(
                    "openjarvis.cli.chat_cmd.record_voice",
                    return_value="what time is it",
                )
            )
            speak = stack.enter_context(patch("openjarvis.cli.chat_cmd.speak"))
            result = CliRunner().invoke(chat, ["--wake", "--model", "test-model"])

        assert result.exit_code == 0
        assert result.exception is None
        # Waited twice: fired the turn, then re-armed and exited.
        assert detector.waits == 2
        record.assert_called_once()
        engine.generate.assert_called_once()
        speak.assert_called_once()
        assert "hi there" in speak.call_args.args[0]
        assert "Hey Jarvis!" in result.output
        assert "hi there" in result.output
        assert "Goodbye!" in result.output

    def test_hey_jarvis_alias_runs_a_turn(self) -> None:
        engine, config = _engine_and_config()
        detector = _FakeDetector([True, False])

        with ExitStack() as stack:
            _enter_base_patches(stack, engine, config, detector)
            stack.enter_context(
                patch("openjarvis.cli.chat_cmd.record_voice", return_value="hello")
            )
            speak = stack.enter_context(patch("openjarvis.cli.chat_cmd.speak"))
            result = CliRunner().invoke(chat, ["--hey-jarvis", "--model", "test-model"])

        assert result.exit_code == 0
        engine.generate.assert_called_once()
        speak.assert_called_once()

    def test_nothing_heard_returns_to_waiting(self) -> None:
        engine, config = _engine_and_config()
        detector = _FakeDetector([True, True, False])

        # First wake yields no transcript (None); second wake yields text.
        with ExitStack() as stack:
            _enter_base_patches(stack, engine, config, detector)
            record = stack.enter_context(
                patch(
                    "openjarvis.cli.chat_cmd.record_voice",
                    side_effect=[None, "hello"],
                )
            )
            speak = stack.enter_context(patch("openjarvis.cli.chat_cmd.speak"))
            result = CliRunner().invoke(chat, ["--wake", "--model", "test-model"])

        assert result.exit_code == 0
        assert record.call_count == 2
        engine.generate.assert_called_once()  # only the turn that heard speech
        speak.assert_called_once()


class TestWakeExit:
    def test_wake_interrupt_exits_before_recording(self) -> None:
        engine, config = _engine_and_config()
        detector = _FakeDetector([False])  # Ctrl+C-equivalent

        with ExitStack() as stack:
            _enter_base_patches(stack, engine, config, detector)
            record = stack.enter_context(patch("openjarvis.cli.chat_cmd.record_voice"))
            speak = stack.enter_context(patch("openjarvis.cli.chat_cmd.speak"))
            result = CliRunner().invoke(chat, ["--wake", "--model", "test-model"])

        assert result.exit_code == 0
        assert result.exception is None
        record.assert_not_called()
        speak.assert_not_called()
        assert "Goodbye!" in result.output

    def test_record_voice_exit_ends_session(self) -> None:
        engine, config = _engine_and_config()
        detector = _FakeDetector([True])

        with ExitStack() as stack:
            _enter_base_patches(stack, engine, config, detector)
            stack.enter_context(
                patch("openjarvis.cli.chat_cmd.record_voice", return_value=VOICE_EXIT)
            )
            speak = stack.enter_context(patch("openjarvis.cli.chat_cmd.speak"))
            result = CliRunner().invoke(chat, ["--wake", "--model", "test-model"])

        assert result.exit_code == 0
        engine.generate.assert_not_called()
        speak.assert_not_called()
        assert "Goodbye!" in result.output

    def test_missing_wake_extra_reports_and_exits(self) -> None:
        engine, config = _engine_and_config()

        detector = MagicMock()
        detector.wait_for_wake.side_effect = RuntimeError(
            "openwakeword is required for the 'Hey Jarvis' wake word. "
            "Install the wake dependencies with: pip install 'OpenJarvis[wake]'"
        )

        with ExitStack() as stack:
            _enter_base_patches(stack, engine, config, detector)
            record = stack.enter_context(patch("openjarvis.cli.chat_cmd.record_voice"))
            result = CliRunner().invoke(chat, ["--wake", "--model", "test-model"])

        assert result.exit_code == 0
        assert result.exception is None
        record.assert_not_called()
        assert "Wake word unavailable" in result.output
        assert "OpenJarvis[wake]" in result.output
