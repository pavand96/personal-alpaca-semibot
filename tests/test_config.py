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
  per_trade_notional: 777
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["watchlist"] == ["NVDA", "ON"]
    assert config["strategy"]["per_trade_notional"] == 777
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
