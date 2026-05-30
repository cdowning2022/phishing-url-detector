"""
Tests for the CLI commands.

We use Typer's CliRunner to invoke the CLI in-process, no subprocess shelling.
These tests require a trained model at models/phishing_model.pkl — if it's
missing, they skip with a clear message rather than fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.predict import app

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "phishing_model.pkl"
runner = CliRunner()


def _require_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"No trained model at {MODEL_PATH}. Run `python -m src.train` first.")


# ---------- info command ----------
class TestInfoCommand:

    def test_info_succeeds(self):
        _require_model()
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0

    def test_info_shows_model_type(self):
        _require_model()
        result = runner.invoke(app, ["info"])
        assert "Model type" in result.stdout

    def test_info_shows_feature_count(self):
        _require_model()
        result = runner.invoke(app, ["info"])
        assert "Feature count" in result.stdout


# ---------- predict command ----------
class TestPredictCommand:

    def test_predict_runs_without_error(self):
        _require_model()
        result = runner.invoke(app, ["predict", "https://example.com"])
        assert result.exit_code == 0

    def test_predict_returns_classification(self):
        _require_model()
        result = runner.invoke(app, ["predict", "https://example.com"])
        # Output should contain one of the two labels
        assert "PHISHING" in result.stdout or "LEGITIMATE" in result.stdout

    def test_predict_shows_confidence(self):
        _require_model()
        result = runner.invoke(app, ["predict", "https://example.com"])
        assert "confidence" in result.stdout.lower()

    def test_verbose_flag_adds_feature_breakdown(self):
        _require_model()
        result = runner.invoke(app, ["predict", "https://example.com", "--verbose"])
        assert result.exit_code == 0
        assert "contributing features" in result.stdout.lower()

    def test_predict_handles_url_without_scheme(self):
        _require_model()
        result = runner.invoke(app, ["predict", "example.com"])
        # Should normalize and run, not crash
        assert result.exit_code == 0


# ---------- error handling ----------
class TestErrorHandling:

    def test_missing_model_fails_gracefully(self, tmp_path):
        fake_path = tmp_path / "does_not_exist.pkl"
        result = runner.invoke(app, ["predict", "https://example.com", "--model", str(fake_path)])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_missing_url_argument(self):
        # No URL → Typer should reject
        result = runner.invoke(app, ["predict"])
        assert result.exit_code != 0
