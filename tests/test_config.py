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


def test_load_config_rejects_ml_per_trade_notional_exceeding_max_position(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
ml:
  per_trade_notional: 500.0
  max_position_notional: 100.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ml.per_trade_notional cannot exceed ml.max_position_notional"):
        load_config(config_path)


def test_load_config_rejects_negative_ml_stop_loss(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
ml:
  stop_loss_pct: -1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stop_loss_pct must be positive"):
        load_config(config_path)


def test_load_config_accepts_max_daily_loss_pct_zero(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
risk:
  max_daily_loss_pct: 0
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config["risk"]["max_daily_loss_pct"] == 0


def test_load_config_rejects_buy_probability_above_1(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
ml:
  buy_probability: 1.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="buy_probability must be between 0 and 1"):
        load_config(config_path)


def test_load_config_rejects_sell_probability_below_0(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
ml:
  sell_probability: -0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sell_probability must be between 0 and 1"):
        load_config(config_path)


def test_load_config_rejects_buy_probability_less_than_sell_probability(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
ml:
  buy_probability: 0.30
  sell_probability: 0.50
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="buy_probability must be >= ml.sell_probability"):
        load_config(config_path)


def test_load_config_accepts_buy_probability_equal_to_sell_probability(tmp_path) -> None:
    # Equal values create a single-threshold mode: buy when flat, sell when held.
    # The two conditions are mutually exclusive within one run (held vs not-held guard).
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
ml:
  buy_probability: 0.50
  sell_probability: 0.50
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config["ml"]["buy_probability"] == 0.50
    assert config["ml"]["sell_probability"] == 0.50


def test_load_config_accepts_max_symbols_to_buy_per_run_zero(tmp_path) -> None:
    # 0 means sell-only mode: existing positions can be sold, no new buys
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
strategy:
  max_symbols_to_buy_per_run: 0
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config["strategy"]["max_symbols_to_buy_per_run"] == 0
