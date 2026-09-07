"""Tests for headed/headless browser resolution and wake greeting helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestResolveHeadless:
    def test_headed_env_forces_visible(self, monkeypatch):
        monkeypatch.setenv("OPENJARVIS_BROWSER_HEADED", "1")
        monkeypatch.delenv("OPENJARVIS_BROWSER_HEADLESS", raising=False)
        from openjarvis.tools.browser import _resolve_headless

        assert _resolve_headless() is False

    def test_headless_env_forces_headless(self, monkeypatch):
        monkeypatch.delenv("OPENJARVIS_BROWSER_HEADED", raising=False)
        monkeypatch.setenv("OPENJARVIS_BROWSER_HEADLESS", "1")
        from openjarvis.tools.browser import _resolve_headless

        assert _resolve_headless() is True

    def test_config_headless_false(self, monkeypatch):
        monkeypatch.delenv("OPENJARVIS_BROWSER_HEADED", raising=False)
        monkeypatch.delenv("OPENJARVIS_BROWSER_HEADLESS", raising=False)
        cfg = MagicMock()
        cfg.tools.browser.headless = False
        with patch("openjarvis.core.config.load_config", return_value=cfg):
            from openjarvis.tools.browser import _resolve_headless

            assert _resolve_headless() is False


class TestEnsureBrowserHeaded:
    def test_launch_passes_headless_false_and_slow_mo(self, monkeypatch):
        """Inject a fake playwright module so we don't need the real package."""
        monkeypatch.setenv("OPENJARVIS_BROWSER_HEADED", "1")
        from openjarvis.tools import browser as browser_mod

        session = browser_mod._BrowserSession()
        fake_pw = MagicMock()
        fake_browser = MagicMock()
        fake_page = MagicMock()
        fake_pw.chromium.launch.return_value = fake_browser
        fake_browser.new_page.return_value = fake_page
        fake_cm = MagicMock()
        fake_cm.start.return_value = fake_pw

        fake_sync_api = MagicMock()
        fake_sync_api.sync_playwright.return_value = fake_cm

        with patch.object(browser_mod, "_resolve_headless", return_value=False):
            with patch.dict(
                "sys.modules",
                {
                    "playwright": MagicMock(),
                    "playwright.sync_api": fake_sync_api,
                },
            ):
                session._ensure_browser()

        kwargs = fake_pw.chromium.launch.call_args.kwargs
        assert kwargs.get("headless") is False
        assert kwargs.get("slow_mo") == 50
        session.close()


class TestWakeGreeting:
    def test_default_greeting(self, monkeypatch):
        monkeypatch.delenv("OPENJARVIS_WAKE_GREETING", raising=False)
        from openjarvis.cli._voice_chat import wake_greeting_text

        assert "Bharth" in wake_greeting_text()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENJARVIS_WAKE_GREETING", "Hello boss")
        from openjarvis.cli._voice_chat import wake_greeting_text

        assert wake_greeting_text() == "Hello boss"
