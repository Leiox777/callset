from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from callset.cli import main, _parse_distribution
from callset.models import ExampleType


FIXTURES = Path(__file__).parent / "fixtures"


class TestParseDistribution:
    def test_valid(self):
        dist = _parse_distribution("happy:40,multi_step:20,clarification:15,error:15,refusal:10")
        assert dist[ExampleType.HAPPY] == 40
        assert sum(dist.values()) == 100

    def test_invalid_type(self):
        with pytest.raises(Exception):
            _parse_distribution("unknown:100")

    def test_does_not_sum_to_100(self):
        with pytest.raises(Exception):
            _parse_distribution("happy:50,multi_step:20")


class TestCLI:
    def test_no_input_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_both_inputs_shows_error(self):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--spec", str(FIXTURES / "sample_openapi.yaml"),
            "--tools", str(FIXTURES / "sample_tools.json"),
        ])
        assert result.exit_code != 0

    def test_dry_run_openapi(self):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--spec", str(FIXTURES / "sample_openapi.yaml"),
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "SkyRoute Travel API" in result.output
        assert "search_flights" in result.output

    def test_dry_run_json_tools(self):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--tools", str(FIXTURES / "sample_tools.json"),
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "search_flights" in result.output
        assert "book_flight" in result.output
