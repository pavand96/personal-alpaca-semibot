import pytest

from semibot.config import load_config


def test_load_config_merges_user_config_and_keeps_dry_run_default(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
alpaca:
  paper: true
watchlist:
  - nvda
  - "ON"
strategy:
  per_trade_notional: 111
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["watchlist"] == ["NVDA", "ON"]
    assert config["strategy"]["per_trade_notional"] == 111
    assert config["risk"]["dry_run"] is True


def test_load_config_rejects_non_string_symbols(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
watchlist:
  - 123
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="watchlist symbols must be strings"):
        load_config(config_path)


def test_load_config_rejects_non_paper_execution_without_dry_run(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
alpaca:
  paper: false
risk:
  dry_run: false
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refusing to load config"):
        load_config(config_path)


def test_load_config_rejects_invalid_time(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
risk:
  exit_before_close: "25:99"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exit_before_close must be HH:MM"):
        load_config(config_path)
