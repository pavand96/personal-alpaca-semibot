from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest, StockSnapshotRequest

from semibot.backtest import (
    Backtester,
    BacktestPosition,
    BacktestResult,
    BacktestTrade,
    DailyBar,
    apply_slippage,
    market_value,
    print_backtest_result,
    write_trades_csv,
)
from semibot.bot import Decision, SemiMomentumBot, format_decision
from semibot.intraday import MARKET_TZ

log = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"


def detect_market_regime(
    risk_bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    sma_fast_days: int,
    sma_slow_days: int,
) -> MarketRegime:
    """3-state regime: BULL if SPY > SMA200 and SMA50 > SMA200, BEAR if both fail, else NEUTRAL."""
    bars_list = list(risk_bars_by_symbol.values())
    if not bars_list:
        return MarketRegime.BULL
    prior = sorted(
        (b for b in bars_list[0] if b.timestamp.date() <= current_date),
        key=lambda b: b.timestamp.date(),
    )
    if len(prior) < sma_slow_days:
        return MarketRegime.NEUTRAL
    sma_slow = sum(b.close for b in prior[-sma_slow_days:]) / sma_slow_days
    sma_fast = sum(b.close for b in prior[-sma_fast_days:]) / sma_fast_days
    above_sma200 = prior[-1].close > sma_slow
    golden_cross = sma_fast > sma_slow
    if above_sma200 and golden_cross:
        return MarketRegime.BULL
    if not above_sma200 and not golden_cross:
        return MarketRegime.BEAR
    return MarketRegime.NEUTRAL


@dataclass(frozen=True)
class AdaptiveCandidate:
    symbol: str
    score: float
    fast_return_pct: float
    slow_return_pct: float
    volume_ratio: float
    sma_alignment: float = 0.0
    relative_strength: float = 0.0
    trend_consistency: float = 0.0
    momentum_acceleration: float = 0.0


@dataclass(frozen=True)
class BuzzSnapshot:
    article_count: int
    positive_hits: int
    negative_hits: int
    score: float


@dataclass(frozen=True)
class ReboundContext:
    active: bool
    sector_drawdown_pct: float
    sector_fast_return_pct: float
    market_fast_return_pct: float
    short_breadth_pct: float
    reason: str


class AdaptiveSemiPortfolioBacktester(Backtester):
    def spike_scan(self, execute: bool = False, already_ordered: set[str] | None = None) -> tuple[list[Decision], set[str]]:
        """Detect pre-market or after-hours spikes and immediately place extended-hours limit orders.

        Works in both windows:
          - Pre-market  (4:00am–9:30am ET): gap vs previous_daily_bar.close (yesterday's 4pm close)
          - After-hours (4:15pm–7:45pm ET): gap vs daily_bar.close (today's 4pm close)

        already_ordered: symbols already entered this session — prevents duplicate orders across
        repeated loop iterations when an order hasn't filled yet.
        Returns (submitted_decisions, updated_already_ordered_set).
        """
        if already_ordered is None:
            already_ordered = set()

        bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
        bot.assert_account_can_trade()
        clock = bot.trading.get_clock()
        if bool(getattr(clock, "is_open", False)):
            print("Market is open — use regular 'run' command for intraday trading.")
            return [], already_ordered

        from semibot.spike_tracker import record_spike_entry

        settings = self.config["adaptive_semis_allocator"]
        min_gap = float(settings.get("spike_min_gap_pct", 5.0))
        max_gap = float(settings.get("spike_max_gap_pct", 20.0))
        notional = float(settings.get("spike_notional_per_trade", 1000.0))
        sector_confirm = int(settings.get("spike_sector_confirm", 1))
        tracker_path = str(settings.get("spike_tracker_path", "logs/spike_tracker.json"))

        gaps = fetch_extended_hours_gaps(bot, self.config["watchlist"])
        if not gaps:
            print("No extended-hours price data available.")
            return [], already_ordered

        # Log every symbol with a move >= 0.5% so we can monitor in the log
        meaningful = [(s, g) for s, g in sorted(gaps.items(), key=lambda x: -abs(x[1])) if abs(g) >= 0.5]
        if meaningful:
            print("  " + "  ".join(f"{s}:{g:+.1f}%" for s, g in meaningful))

        qualifying = {s: g for s, g in gaps.items() if min_gap <= g <= max_gap}
        if not qualifying:
            top = max(gaps.values(), default=0.0)
            print(f"No spikes >= {min_gap:.1f}% (best: {top:+.1f}%)")
            return [], already_ordered

        if len(qualifying) < sector_confirm:
            print(f"Only {len(qualifying)} symbol(s) gapping — need {sector_confirm} for sector confirmation. Skipping.")
            return [], already_ordered

        positions = bot.get_positions()
        held = {s for s, p in positions.items() if float(getattr(p, "qty", 0)) > 0}
        live_prices = fetch_live_prices(bot, list(qualifying.keys()))

        decisions: list[Decision] = []
        for symbol, gap_pct in sorted(qualifying.items(), key=lambda x: -x[1]):
            if symbol in held:
                continue
            if symbol in already_ordered:
                print(f"  ALREADY ORDERED {symbol} ({gap_pct:+.1f}%) — skipping duplicate")
                continue
            print(f"  *** SPIKE {symbol}: {gap_pct:+.1f}% vs last close — placing extended-hours order ***")
            already_ordered.add(symbol)
            decisions.append(
                Decision(symbol, "buy", f"spike gap={gap_pct:+.1f}%", notional=notional)
            )

        dry_run = bool(self.config["risk"]["dry_run"]) or not execute
        submitted = submit_adaptive_decisions(
            bot=bot,
            decisions=decisions,
            dry_run=dry_run,
            max_orders=int(self.config["risk"]["max_orders_per_run"]),
            extended_hours=True,
        )
        if decisions and not submitted:
            print("No spike orders placed.")
        if not dry_run:
            for decision in submitted:
                if decision.action == "buy":
                    entry_price = live_prices.get(decision.symbol, 0.0)
                    try:
                        record_spike_entry(tracker_path, decision.symbol, qualifying[decision.symbol], entry_price)
                    except Exception as exc:
                        log.warning("spike_tracker record failed for %s: %s", decision.symbol, exc)
        return submitted, already_ordered

    # backward-compat alias
    def afterhours_scan(self, execute: bool = False) -> list[Decision]:
        submitted, _ = self.spike_scan(execute=execute)
        return submitted

    def trade_once(self, execute: bool = False, premarket: bool = False) -> list[Decision]:
        bot = SemiMomentumBot(self.config, api_key=self.api_key, secret_key=self.secret_key)
        bot.assert_account_can_trade()
        clock = bot.trading.get_clock()
        market_open = bool(getattr(clock, "is_open", False))
        if not premarket and self.config["risk"]["require_market_open"] and not market_open:
            print("Market is closed. No adaptive orders submitted.")
            return []
        if premarket and market_open:
            print("Market is open — use regular 'run' command instead of premarket.")
            return []

        positions = bot.get_positions()
        if bot.daily_loss_kill_switch_triggered():
            if self.config["risk"].get("flatten_on_daily_loss", True):
                decisions = [
                    Decision(position.symbol, "sell", "daily loss kill switch", qty=float(position.qty))
                    for position in positions.values()
                    if float(position.qty) > 0
                ]
                return submit_adaptive_decisions(
                    bot=bot,
                    decisions=decisions,
                    dry_run=bool(self.config["risk"]["dry_run"]) or not execute,
                    max_orders=len(decisions),
                )
            print("Daily loss kill switch active. No adaptive orders submitted.")
            return []

        decisions = self.live_decisions(bot=bot, positions=positions, premarket=premarket)
        submitted = submit_adaptive_decisions(
            bot=bot,
            decisions=decisions,
            dry_run=bool(self.config["risk"]["dry_run"]) or not execute,
            max_orders=int(self.config["risk"]["max_orders_per_run"]),
            extended_hours=premarket,
        )
        if not submitted:
            print("No adaptive trade signals.")
        return submitted

    def live_decisions(self, bot: SemiMomentumBot, positions: dict[str, Any], premarket: bool = False) -> list[Decision]:
        settings = self.config["adaptive_semis_allocator"]
        overlay_settings = self.config.get("buzz_earnings_overlay", {})
        current_date = datetime.now(MARKET_TZ).date()

        max_symbols = int(settings["max_symbols"])
        max_total_exposure = float(settings["max_total_exposure"])
        fast_lookback = int(settings["fast_momentum_days"])
        slow_lookback = int(settings["momentum_lookback_days"])
        sector_lookback = int(settings["sector_lookback_days"])
        sector_slow_lookback = int(settings["sector_slow_lookback_days"])
        sector_drawdown_lookback = int(settings["sector_drawdown_lookback_days"])
        min_sector_return = float(settings["min_sector_return_pct"])
        min_sector_slow_return = float(settings["min_sector_slow_return_pct"])
        risk_off_sector_return = float(settings["risk_off_sector_return_pct"])
        max_sector_drawdown = float(settings["max_sector_drawdown_pct"])
        risk_filter_symbols = [str(symbol).upper() for symbol in settings.get("risk_filter_symbols", [])]
        market_momentum_lookback = int(settings["market_momentum_lookback_days"])
        market_sma_days = int(settings["market_sma_days"])
        min_market_momentum = float(settings["min_market_momentum_pct"])
        require_market_above_sma = bool(settings["require_market_above_sma"])
        min_symbol_return = float(settings["min_symbol_momentum_pct"])
        min_trade_notional = float(settings["min_trade_notional"])
        retain_winners = bool(settings["retain_winners"])
        retain_min_profit_pct = float(settings["retain_min_profit_pct"])
        retain_min_fast_momentum_pct = float(settings["retain_min_fast_momentum_pct"])
        retain_min_slow_momentum_pct = float(settings["retain_min_slow_momentum_pct"])
        hard_stop_pct = float(settings["hard_stop_pct"])
        regime_filter_enabled = bool(settings.get("regime_filter_enabled", False))
        regime_sma_fast = int(settings.get("regime_sma_fast_days", 50))
        regime_sma_slow = int(settings.get("regime_sma_slow_days", 200))
        neutral_sizing_pct = float(settings.get("neutral_sizing_pct", 1.0))
        bear_block_entries = bool(settings.get("bear_block_entries", True))
        symbol_sma_filter_days = int(settings.get("symbol_sma_filter_days", 0))
        use_sma_alignment = bool(settings.get("use_sma_alignment", False))
        use_relative_strength = bool(settings.get("use_relative_strength", False))
        use_trend_consistency = bool(settings.get("use_trend_consistency", False))
        use_momentum_acceleration = bool(settings.get("use_momentum_acceleration", False))
        momentum_acceleration_weight = float(settings.get("momentum_acceleration_weight", 6.0))
        slow_momentum_weight = float(settings.get("slow_momentum_weight", 0.9))
        fast_momentum_weight = float(settings.get("fast_momentum_weight", 1.8))
        short_momentum_weight = float(settings.get("short_momentum_weight", 0.6))
        volume_weight = float(settings.get("volume_weight", 2.0))
        sector_weight_score = float(settings.get("sector_weight_score", 0.5))
        high_proximity_weight = float(settings.get("high_proximity_weight", 15.0))
        sma_alignment_weight = float(settings.get("sma_alignment_weight", 12.0))
        trend_consistency_weight = float(settings.get("trend_consistency_weight", 8.0))
        gap_boost_weight = float(settings.get("gap_boost_weight", 0.0))
        gap_boost_min_pct = float(settings.get("gap_boost_min_pct", 2.0))
        score_proportional_sizing = bool(settings.get("score_proportional_sizing", False))
        bull_exposure_pct = float(settings.get("bull_exposure_pct", 0.0))
        bull_max_symbols = int(settings.get("bull_max_symbols", max_symbols))
        bull_sector_min_return = float(settings.get("bull_sector_min_return_pct", 0.0))
        bull_sector_fast_lookback = int(settings.get("bull_sector_fast_lookback_days", 5))
        bull_sector_fast_min_return = float(settings.get("bull_sector_fast_min_return_pct", -999.0))
        atr_sizing_enabled = bool(settings.get("atr_sizing_enabled", False))
        atr_lookback_days = int(settings.get("atr_lookback_days", 14))
        breadth_filter_enabled = bool(settings.get("breadth_filter_enabled", False))
        breadth_sma_days = int(settings.get("breadth_sma_days", 50))
        min_breadth_pct = float(settings.get("min_breadth_pct", 0.5))

        max_lookback = max(
            fast_lookback,
            slow_lookback,
            sector_lookback,
            sector_slow_lookback,
            sector_drawdown_lookback,
            market_momentum_lookback,
            market_sma_days,
            regime_sma_slow if regime_filter_enabled else 0,
            symbol_sma_filter_days,
            atr_lookback_days + 1,
            breadth_sma_days if breadth_filter_enabled else 0,
            int(settings.get("rebound_fast_lookback_days", 5)) if settings.get("rebound_mode_enabled", False) else 0,
            int(settings.get("rebound_breadth_sma_days", 5)) if settings.get("rebound_mode_enabled", False) else 0,
            int(settings.get("rebound_symbol_lookback_days", 3)) if settings.get("rebound_mode_enabled", False) else 0,
            50,
            30,
        )
        fetch_start = current_date - timedelta(days=max_lookback * 3)
        bars_by_symbol = self.fetch_daily_bars(self.config["watchlist"], fetch_start, current_date)
        risk_bars_by_symbol = self.fetch_daily_bars(risk_filter_symbols, fetch_start, current_date) if risk_filter_symbols else {}
        latest_prices = fetch_live_prices(bot, self.config["watchlist"])
        overlay_bars = live_overlay_bars(bars_by_symbol=bars_by_symbol, prices=latest_prices, current_date=current_date)
        # Always compute gaps in premarket mode (for filtering); otherwise only when boost is active
        gap_by_symbol = _compute_gap_by_symbol(latest_prices, bars_by_symbol) if (gap_boost_weight > 0 or premarket) else None
        overlay_positions = backtest_positions_from_live(self.config["watchlist"], positions)

        news_by_symbol = fetch_news_by_symbol(
            api_key=self.api_key,
            secret_key=self.secret_key,
            symbols=self.config["watchlist"],
            start=current_date - timedelta(days=int(overlay_settings.get("news_lookback_days", 7))),
            end=current_date,
            settings=overlay_settings,
        )
        earnings_by_symbol = load_earnings_calendar(overlay_settings.get("earnings_calendar_file"))

        sector_return = sector_return_pct(bars_by_symbol, current_date, sector_lookback)
        sector_slow_return = sector_return_pct(bars_by_symbol, current_date, sector_slow_lookback)
        sector_drawdown = sector_drawdown_pct(bars_by_symbol, current_date, sector_drawdown_lookback)
        market_ok = market_filter_allows_entries(
            bars_by_symbol=risk_bars_by_symbol,
            current_date=current_date,
            momentum_lookback=market_momentum_lookback,
            sma_days=market_sma_days,
            min_momentum_pct=min_market_momentum,
            require_above_sma=require_market_above_sma,
        )
        breadth = sector_breadth_pct(bars_by_symbol, current_date, breadth_sma_days) if breadth_filter_enabled else 1.0
        risk_off = (
            sector_return <= risk_off_sector_return
            or sector_slow_return < min_sector_slow_return
            or sector_drawdown <= max_sector_drawdown
            or not market_ok
            or (breadth_filter_enabled and breadth < min_breadth_pct)
        )
        regime = (
            detect_market_regime(risk_bars_by_symbol, current_date, regime_sma_fast, regime_sma_slow)
            if regime_filter_enabled
            else MarketRegime.BULL
        )
        rebound = detect_rebound_context(bars_by_symbol, risk_bars_by_symbol, current_date, settings)

        candidates = rank_adaptive_candidates(
            bars_by_symbol=bars_by_symbol,
            current_date=current_date,
            fast_lookback=fast_lookback,
            slow_lookback=slow_lookback,
            sector_lookback=sector_lookback,
            min_sector_return=min_sector_return,
            min_symbol_return=min_symbol_return,
            symbol_sma_filter_days=symbol_sma_filter_days,
            use_sma_alignment=use_sma_alignment,
            use_relative_strength=use_relative_strength,
            use_trend_consistency=use_trend_consistency,
            use_momentum_acceleration=use_momentum_acceleration,
            momentum_acceleration_weight=momentum_acceleration_weight,
            slow_momentum_weight=slow_momentum_weight,
            fast_momentum_weight=fast_momentum_weight,
            short_momentum_weight=short_momentum_weight,
            volume_weight=volume_weight,
            sector_weight_score=sector_weight_score,
            high_proximity_weight=high_proximity_weight,
            sma_alignment_weight=sma_alignment_weight,
            trend_consistency_weight=trend_consistency_weight,
            gap_by_symbol=gap_by_symbol,
            gap_boost_weight=gap_boost_weight,
            gap_boost_min_pct=gap_boost_min_pct,
        )
        candidates = apply_buzz_earnings_overlay(
            candidates=candidates,
            current_date=current_date,
            positions=overlay_positions,
            today_bars=overlay_bars,
            news_by_symbol=news_by_symbol,
            earnings_by_symbol=earnings_by_symbol,
            settings=overlay_settings,
        )
        bear_blocked = regime_filter_enabled and regime == MarketRegime.BEAR and bear_block_entries
        sector_fast_return = sector_return_pct(bars_by_symbol, current_date, bull_sector_fast_lookback)
        live_use_full_bull = (
            regime_filter_enabled
            and regime == MarketRegime.BULL
            and bull_exposure_pct > 0
            and sector_return >= bull_sector_min_return
            and sector_fast_return >= bull_sector_fast_min_return
        )
        if rebound.active:
            rebound_candidates = rank_rebound_candidates(bars_by_symbol, current_date, settings)
            rebound_candidates = apply_buzz_earnings_overlay(
                candidates=rebound_candidates,
                current_date=current_date,
                positions=overlay_positions,
                today_bars=overlay_bars,
                news_by_symbol=news_by_symbol,
                earnings_by_symbol=earnings_by_symbol,
                settings=overlay_settings,
            )
            effective_max_symbols_live = int(settings.get("rebound_max_symbols", max_symbols))
            selected = rebound_candidates[:effective_max_symbols_live]
        else:
            effective_max_symbols_live = bull_max_symbols if live_use_full_bull else max_symbols
            selected = [] if (risk_off or bear_blocked) else candidates[:effective_max_symbols_live]
        target_symbols = {candidate.symbol for candidate in selected}

        from semibot.spike_tracker import get_symbols_to_exit, remove_spike_entry
        tracker_path = str(settings.get("spike_tracker_path", "logs/spike_tracker.json"))
        spike_exit_symbols = set(get_symbols_to_exit(tracker_path, current_date))

        decisions: list[Decision] = []
        held_symbols = set()
        for symbol in self.config["watchlist"]:
            position = positions.get(symbol)
            if not position:
                if symbol in spike_exit_symbols:
                    remove_spike_entry(tracker_path, symbol)  # stale — position already gone
                continue
            qty = float(getattr(position, "qty", 0.0) or 0.0)
            if qty <= 0:
                if symbol in spike_exit_symbols:
                    remove_spike_entry(tracker_path, symbol)
                continue
            held_symbols.add(symbol)
            current_price = latest_prices.get(symbol)
            unrealized_plpc = float(getattr(position, "unrealized_plpc", 0.0) or 0.0)
            if unrealized_plpc <= -(hard_stop_pct / 100):
                if symbol in spike_exit_symbols:
                    remove_spike_entry(tracker_path, symbol)
                decisions.append(
                    Decision(symbol, "sell", f"adaptive hard stop {unrealized_plpc:.2%} <= -{hard_stop_pct:.2f}%", qty=qty)
                )
                continue
            if risk_off and not rebound.active:
                if symbol in spike_exit_symbols:
                    remove_spike_entry(tracker_path, symbol)
                decisions.append(
                    Decision(
                        symbol,
                        "sell",
                        (
                            "adaptive risk-off: "
                            f"sector={sector_return:.2f}% slow={sector_slow_return:.2f}% "
                            f"dd={sector_drawdown:.2f}% market_ok={market_ok}"
                        ),
                        qty=qty,
                    )
                )
                continue
            # Force-exit spike positions that have held for 1 session (entry_date < today)
            if symbol in spike_exit_symbols:
                remove_spike_entry(tracker_path, symbol)
                decisions.append(Decision(symbol, "sell", "spike 1-day hold exit", qty=qty))
                continue
            if symbol in target_symbols:
                continue
            if (
                retain_winners
                and current_price
                and should_retain_winner(
                    symbol=symbol,
                    position=overlay_positions[symbol],
                    current_price=current_price,
                    bars_by_symbol=bars_by_symbol,
                    current_date=current_date,
                    fast_lookback=fast_lookback,
                    slow_lookback=slow_lookback,
                    min_profit_pct=retain_min_profit_pct,
                    min_fast_momentum_pct=retain_min_fast_momentum_pct,
                    min_slow_momentum_pct=retain_min_slow_momentum_pct,
                )
            ):
                decisions.append(Decision(symbol, "hold", "adaptive retained winner"))
                continue
            decisions.append(Decision(symbol, "sell", "adaptive rebalance out", qty=qty))

        if (risk_off and not rebound.active) or (bear_blocked and not rebound.active) or not selected:
            return decisions

        account = bot.trading.get_account()
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
        if rebound.active:
            total_pool = equity * float(settings.get("rebound_exposure_pct", 0.55))
        elif live_use_full_bull:
            total_pool = equity * bull_exposure_pct
        else:
            regime_mult = neutral_sizing_pct if (regime_filter_enabled and regime == MarketRegime.NEUTRAL) else 1.0
            total_pool = min(max_total_exposure, equity) * regime_mult
        if atr_sizing_enabled:
            alloc_weights = compute_atr_weights(selected, bars_by_symbol, current_date, atr_lookback_days)
        elif score_proportional_sizing and len(selected) > 1:
            min_score = min(c.score for c in selected)
            adj_scores = [c.score - min_score + 1.0 for c in selected]
            total_adj = sum(adj_scores)
            alloc_weights = [s / total_adj for s in adj_scores]
        else:
            alloc_weights = [1.0 / len(selected)] * len(selected)
        queued_buys = 0.0
        for idx, candidate in enumerate(selected):
            position = positions.get(candidate.symbol)
            held_value = abs(float(getattr(position, "market_value", 0.0) or 0.0)) if position else 0.0
            buy_notional = max(0.0, min(total_pool * alloc_weights[idx] - held_value, buying_power - queued_buys))
            if buy_notional < min_trade_notional:
                continue
            # In pre-market mode, only buy symbols with a confirmed large gap
            if premarket and gap_by_symbol is not None:
                premarket_min_gap = float(settings.get("premarket_min_gap_pct", 4.0))
                premarket_max_gap = float(settings.get("premarket_max_gap_pct", 20.0))
                gap = gap_by_symbol.get(candidate.symbol, 0.0)
                if not (premarket_min_gap <= gap <= premarket_max_gap):
                    print(f"PRE-MARKET SKIP {candidate.symbol}: gap={gap:.1f}% (need {premarket_min_gap:.1f}%–{premarket_max_gap:.1f}%)")
                    continue

            queued_buys += buy_notional
            gap_note = ""
            if gap_by_symbol:
                gap_pct = gap_by_symbol.get(candidate.symbol, 0.0)
                gap_note = f" gap={gap_pct:+.1f}%"
            decisions.append(
                Decision(
                    candidate.symbol,
                    "buy",
                    (
                        f"{'rebound' if rebound.active else 'adaptive'} target score={candidate.score:.2f}{gap_note}; "
                        f"sector={sector_return:.2f}% slow={sector_slow_return:.2f}% dd={sector_drawdown:.2f}%"
                    ),
                    notional=buy_notional,
                )
            )

        return decisions

    def run(self, start: date, end: date) -> BacktestResult:
        settings = self.config["adaptive_semis_allocator"]
        overlay_settings = self.config.get("buzz_earnings_overlay", {})
        starting_cash = float(self.config["backtest"]["initial_cash"])
        cash = starting_cash
        positions = {symbol: BacktestPosition() for symbol in self.config["watchlist"]}
        trades: list[BacktestTrade] = []
        per_symbol_pnl = {symbol: 0.0 for symbol in self.config["watchlist"]}

        max_symbols = int(settings["max_symbols"])
        max_total_exposure = float(settings["max_total_exposure"])
        rebalance_days = int(settings["rebalance_days"])
        fast_lookback = int(settings["fast_momentum_days"])
        slow_lookback = int(settings["momentum_lookback_days"])
        sector_lookback = int(settings["sector_lookback_days"])
        sector_slow_lookback = int(settings["sector_slow_lookback_days"])
        sector_drawdown_lookback = int(settings["sector_drawdown_lookback_days"])
        min_sector_return = float(settings["min_sector_return_pct"])
        min_sector_slow_return = float(settings["min_sector_slow_return_pct"])
        risk_off_sector_return = float(settings["risk_off_sector_return_pct"])
        max_sector_drawdown = float(settings["max_sector_drawdown_pct"])
        risk_filter_symbols = [str(symbol).upper() for symbol in settings.get("risk_filter_symbols", [])]
        market_momentum_lookback = int(settings["market_momentum_lookback_days"])
        market_sma_days = int(settings["market_sma_days"])
        min_market_momentum = float(settings["min_market_momentum_pct"])
        require_market_above_sma = bool(settings["require_market_above_sma"])
        min_symbol_return = float(settings["min_symbol_momentum_pct"])
        min_trade_notional = float(settings["min_trade_notional"])
        retain_winners = bool(settings["retain_winners"])
        retain_min_profit_pct = float(settings["retain_min_profit_pct"])
        retain_min_fast_momentum_pct = float(settings["retain_min_fast_momentum_pct"])
        retain_min_slow_momentum_pct = float(settings["retain_min_slow_momentum_pct"])
        trailing_stop_pct = float(settings["trailing_stop_pct"]) / 100
        hard_stop_pct = float(settings["hard_stop_pct"]) / 100
        trailing_stop_pct_bull = float(settings.get("trailing_stop_pct_bull", settings["trailing_stop_pct"])) / 100
        trailing_wide_min_profit = float(settings.get("trailing_wide_min_profit_pct", 8.0)) / 100
        slippage_bps = float(self.config["backtest"]["slippage_bps"])
        regime_filter_enabled = bool(settings.get("regime_filter_enabled", False))
        regime_sma_fast = int(settings.get("regime_sma_fast_days", 50))
        regime_sma_slow = int(settings.get("regime_sma_slow_days", 200))
        neutral_sizing_pct = float(settings.get("neutral_sizing_pct", 1.0))
        bear_block_entries = bool(settings.get("bear_block_entries", True))
        symbol_sma_filter_days = int(settings.get("symbol_sma_filter_days", 0))
        use_sma_alignment = bool(settings.get("use_sma_alignment", False))
        use_relative_strength = bool(settings.get("use_relative_strength", False))
        use_trend_consistency = bool(settings.get("use_trend_consistency", False))
        use_momentum_acceleration = bool(settings.get("use_momentum_acceleration", False))
        momentum_acceleration_weight = float(settings.get("momentum_acceleration_weight", 6.0))
        slow_momentum_weight = float(settings.get("slow_momentum_weight", 0.9))
        fast_momentum_weight = float(settings.get("fast_momentum_weight", 1.8))
        short_momentum_weight = float(settings.get("short_momentum_weight", 0.6))
        volume_weight = float(settings.get("volume_weight", 2.0))
        sector_weight_score = float(settings.get("sector_weight_score", 0.5))
        high_proximity_weight = float(settings.get("high_proximity_weight", 15.0))
        sma_alignment_weight = float(settings.get("sma_alignment_weight", 12.0))
        trend_consistency_weight = float(settings.get("trend_consistency_weight", 8.0))
        score_proportional_sizing = bool(settings.get("score_proportional_sizing", False))
        bull_exposure_pct = float(settings.get("bull_exposure_pct", 0.0))
        bull_max_symbols = int(settings.get("bull_max_symbols", max_symbols))
        rebalance_days_bull = int(settings.get("rebalance_days_bull", rebalance_days))
        bull_sector_min_return = float(settings.get("bull_sector_min_return_pct", 0.0))
        bull_sector_fast_lookback = int(settings.get("bull_sector_fast_lookback_days", 5))
        bull_sector_fast_min_return = float(settings.get("bull_sector_fast_min_return_pct", -999.0))
        bull_drawdown_cap = float(settings.get("bull_drawdown_cap_pct", 0.0)) / 100
        atr_sizing_enabled = bool(settings.get("atr_sizing_enabled", False))
        atr_lookback_days = int(settings.get("atr_lookback_days", 14))
        breadth_filter_enabled = bool(settings.get("breadth_filter_enabled", False))
        breadth_sma_days = int(settings.get("breadth_sma_days", 50))
        min_breadth_pct = float(settings.get("min_breadth_pct", 0.5))

        max_lookback = max(
            fast_lookback,
            slow_lookback,
            sector_lookback,
            sector_slow_lookback,
            sector_drawdown_lookback,
            market_momentum_lookback,
            market_sma_days,
            regime_sma_slow if regime_filter_enabled else 0,
            symbol_sma_filter_days,
            atr_lookback_days + 1,
            breadth_sma_days if breadth_filter_enabled else 0,
            int(settings.get("rebound_fast_lookback_days", 5)) if settings.get("rebound_mode_enabled", False) else 0,
            int(settings.get("rebound_breadth_sma_days", 5)) if settings.get("rebound_mode_enabled", False) else 0,
            int(settings.get("rebound_symbol_lookback_days", 3)) if settings.get("rebound_mode_enabled", False) else 0,
            50,
            30,
        )
        fetch_start = start - timedelta(days=max_lookback * 3)
        bars_by_symbol = self.fetch_daily_bars(self.config["watchlist"], fetch_start, end)
        risk_bars_by_symbol = self.fetch_daily_bars(risk_filter_symbols, fetch_start, end) if risk_filter_symbols else {}
        news_by_symbol = fetch_news_by_symbol(
            api_key=self.api_key,
            secret_key=self.secret_key,
            symbols=self.config["watchlist"],
            start=start - timedelta(days=int(overlay_settings.get("news_lookback_days", 7))),
            end=end,
            settings=overlay_settings,
        )
        earnings_by_symbol = load_earnings_calendar(overlay_settings.get("earnings_calendar_file"))
        dates = sorted({bar.timestamp.date() for bars in bars_by_symbol.values() for bar in bars})
        bars_by_date = index_daily_bars_by_date(bars_by_symbol)

        last_rebalance_index = -rebalance_days
        peak_equity = starting_cash
        max_drawdown_pct = 0.0

        for day_index, current_date in enumerate(dates):
            if current_date < start or current_date > end:
                continue
            today_bars = bars_by_date[current_date]

            sector_return = sector_return_pct(bars_by_symbol, current_date, sector_lookback)
            sector_slow_return = sector_return_pct(bars_by_symbol, current_date, sector_slow_lookback)
            sector_drawdown = sector_drawdown_pct(bars_by_symbol, current_date, sector_drawdown_lookback)
            market_ok = market_filter_allows_entries(
                bars_by_symbol=risk_bars_by_symbol,
                current_date=current_date,
                momentum_lookback=market_momentum_lookback,
                sma_days=market_sma_days,
                min_momentum_pct=min_market_momentum,
                require_above_sma=require_market_above_sma,
            )
            breadth = sector_breadth_pct(bars_by_symbol, current_date, breadth_sma_days) if breadth_filter_enabled else 1.0
            regime = (
                detect_market_regime(risk_bars_by_symbol, current_date, regime_sma_fast, regime_sma_slow)
                if regime_filter_enabled
                else MarketRegime.BULL
            )
            rebound = detect_rebound_context(bars_by_symbol, risk_bars_by_symbol, current_date, settings)
            for symbol, position in positions.items():
                if position.qty <= 0 or symbol not in today_bars:
                    continue
                bar = today_bars[symbol]
                hard_stop_price = position.avg_entry * (1 - hard_stop_pct)
                # Widen trailing stop in BULL regime once position has earned a profit cushion
                unrealized_pct = (bar.open / position.avg_entry - 1) if position.avg_entry > 0 else 0.0
                in_bull = regime_filter_enabled and regime == MarketRegime.BULL
                effective_trailing = (
                    trailing_stop_pct_bull
                    if in_bull and unrealized_pct >= trailing_wide_min_profit
                    else trailing_stop_pct
                )
                trailing_stop_price = position.peak_price * (1 - effective_trailing)
                stop_price = max(hard_stop_price, trailing_stop_price)
                if bar.open <= stop_price:
                    cash = sell_position(
                        trades=trades,
                        per_symbol_pnl=per_symbol_pnl,
                        cash=cash,
                        position=position,
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        price=apply_slippage(bar.open, "sell", slippage_bps),
                        reason="gap below adaptive stop",
                    )
                    continue
                if bar.low <= stop_price:
                    cash = sell_position(
                        trades=trades,
                        per_symbol_pnl=per_symbol_pnl,
                        cash=cash,
                        position=position,
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        price=apply_slippage(stop_price, "sell", slippage_bps),
                        reason="adaptive trailing/hard stop",
                    )
                    continue
                position.peak_price = max(position.peak_price, bar.high)

            current_equity = cash + market_value(positions, today_bars)
            portfolio_dd = (current_equity / peak_equity - 1) if peak_equity > 0 else 0.0
            drawdown_breaker_tripped = bull_drawdown_cap > 0 and portfolio_dd <= -bull_drawdown_cap
            sector_fast_return = sector_return_pct(bars_by_symbol, current_date, bull_sector_fast_lookback)
            use_full_bull = (
                regime_filter_enabled
                and regime == MarketRegime.BULL
                and bull_exposure_pct > 0
                and sector_return >= bull_sector_min_return
                and sector_fast_return >= bull_sector_fast_min_return
                and not drawdown_breaker_tripped
                and not rebound.active
            )
            effective_rebalance_days = (
                int(settings.get("rebound_rebalance_days", rebalance_days))
                if rebound.active
                else rebalance_days_bull if use_full_bull else rebalance_days
            )
            effective_max_symbols = (
                int(settings.get("rebound_max_symbols", max_symbols))
                if rebound.active
                else bull_max_symbols if use_full_bull else max_symbols
            )
            should_rebalance = day_index - last_rebalance_index >= effective_rebalance_days
            if should_rebalance:
                if rebound.active:
                    candidates = rank_rebound_candidates(
                        bars_by_symbol=bars_by_symbol,
                        current_date=current_date,
                        settings=settings,
                    )
                else:
                    candidates = rank_adaptive_candidates(
                        bars_by_symbol=bars_by_symbol,
                        current_date=current_date,
                        fast_lookback=fast_lookback,
                        slow_lookback=slow_lookback,
                        sector_lookback=sector_lookback,
                        min_sector_return=min_sector_return,
                        min_symbol_return=min_symbol_return,
                        symbol_sma_filter_days=symbol_sma_filter_days,
                        use_sma_alignment=use_sma_alignment,
                        use_relative_strength=use_relative_strength,
                        use_trend_consistency=use_trend_consistency,
                        use_momentum_acceleration=use_momentum_acceleration,
                        momentum_acceleration_weight=momentum_acceleration_weight,
                        slow_momentum_weight=slow_momentum_weight,
                        fast_momentum_weight=fast_momentum_weight,
                        short_momentum_weight=short_momentum_weight,
                        volume_weight=volume_weight,
                        sector_weight_score=sector_weight_score,
                        high_proximity_weight=high_proximity_weight,
                        sma_alignment_weight=sma_alignment_weight,
                        trend_consistency_weight=trend_consistency_weight,
                    )
                candidates = apply_buzz_earnings_overlay(
                    candidates=candidates,
                    current_date=current_date,
                    positions=positions,
                    today_bars=today_bars,
                    news_by_symbol=news_by_symbol,
                    earnings_by_symbol=earnings_by_symbol,
                    settings=overlay_settings,
                )
                risk_off = (
                    sector_return <= risk_off_sector_return
                    or sector_slow_return < min_sector_slow_return
                    or sector_drawdown <= max_sector_drawdown
                    or not market_ok
                    or (breadth_filter_enabled and breadth < min_breadth_pct)
                )
                bear_blocked = regime_filter_enabled and regime == MarketRegime.BEAR and bear_block_entries
                target_symbols = (
                    set() if ((risk_off or bear_blocked) and not rebound.active)
                    else {candidate.symbol for candidate in candidates[:effective_max_symbols]}
                )

                for symbol, position in positions.items():
                    if position.qty <= 0 or symbol in target_symbols or symbol not in today_bars:
                        continue
                    if (
                        retain_winners
                        and (not risk_off or rebound.active)
                        and (not bear_blocked or rebound.active)
                        and should_retain_winner(
                            symbol=symbol,
                            position=position,
                            current_price=today_bars[symbol].close,
                            bars_by_symbol=bars_by_symbol,
                            current_date=current_date,
                            fast_lookback=fast_lookback,
                            slow_lookback=slow_lookback,
                            min_profit_pct=retain_min_profit_pct,
                            min_fast_momentum_pct=retain_min_fast_momentum_pct,
                            min_slow_momentum_pct=retain_min_slow_momentum_pct,
                        )
                    ):
                        continue
                    bar = today_bars[symbol]
                    cash = sell_position(
                        trades=trades,
                        per_symbol_pnl=per_symbol_pnl,
                        cash=cash,
                        position=position,
                        symbol=symbol,
                        timestamp=bar.timestamp,
                        price=apply_slippage(bar.open, "sell", slippage_bps),
                        reason=(
                            "rebound rebalance out"
                            if rebound.active
                            else
                            "adaptive risk-off rebalance to cash"
                            if risk_off
                            else "adaptive rebalance out"
                        ),
                    )

                selected = candidates[:effective_max_symbols] if not ((risk_off or bear_blocked) and not rebound.active) else []
                if selected:
                    if rebound.active:
                        total_pool = current_equity * float(settings.get("rebound_exposure_pct", 0.55))
                    elif use_full_bull:
                        total_pool = current_equity * bull_exposure_pct
                    else:
                        regime_mult = neutral_sizing_pct if (regime_filter_enabled and regime == MarketRegime.NEUTRAL) else 1.0
                        total_pool = min(max_total_exposure, current_equity) * regime_mult
                    if atr_sizing_enabled:
                        alloc_weights = compute_atr_weights(selected, bars_by_symbol, current_date, atr_lookback_days)
                    elif score_proportional_sizing and len(selected) > 1:
                        min_score = min(c.score for c in selected)
                        adj_scores = [c.score - min_score + 1.0 for c in selected]
                        total_adj = sum(adj_scores)
                        alloc_weights = [s / total_adj for s in adj_scores]
                    else:
                        alloc_weights = [1.0 / len(selected)] * len(selected)
                    for idx, candidate in enumerate(selected):
                        if candidate.symbol not in today_bars:
                            continue
                        position = positions[candidate.symbol]
                        bar = today_bars[candidate.symbol]
                        price = apply_slippage(bar.open, "buy", slippage_bps)
                        held_value = position.qty * price
                        buy_notional = max(0.0, min(total_pool * alloc_weights[idx] - held_value, cash))
                        if buy_notional < min_trade_notional:
                            continue
                        qty = buy_notional / price
                        previous_cost = position.avg_entry * position.qty
                        position.qty += qty
                        position.avg_entry = (previous_cost + buy_notional) / position.qty
                        position.peak_price = max(position.peak_price, bar.high, price)
                        cash -= buy_notional
                        trades.append(
                            BacktestTrade(
                                timestamp=bar.timestamp,
                                symbol=candidate.symbol,
                                action="buy",
                                qty=qty,
                                price=price,
                                notional=buy_notional,
                                cash_after=cash,
                                reason=(
                                    f"{'rebound' if rebound.active else 'adaptive'} score={candidate.score:.2f} "
                                    f"fast={candidate.fast_return_pct:.2f}% slow={candidate.slow_return_pct:.2f}% "
                                    f"volume={candidate.volume_ratio:.2f}x sector={sector_return:.2f}% "
                                    f"sector_slow={sector_slow_return:.2f}% sector_dd={sector_drawdown:.2f}%"
                                ),
                            )
                        )
                last_rebalance_index = day_index

            equity = cash + market_value(positions, today_bars)
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                drawdown_pct = ((equity - peak_equity) / peak_equity) * 100
                max_drawdown_pct = min(max_drawdown_pct, drawdown_pct)

        if dates:
            final_bars = bars_by_date[dates[-1]]
            for symbol, position in positions.items():
                if position.qty <= 0 or symbol not in final_bars:
                    continue
                bar = final_bars[symbol]
                cash = sell_position(
                    trades=trades,
                    per_symbol_pnl=per_symbol_pnl,
                    cash=cash,
                    position=position,
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    price=apply_slippage(bar.close, "sell", slippage_bps),
                    reason="liquidate at end",
                )

        ending_equity = cash
        total_return_pct = ((ending_equity - starting_cash) / starting_cash) * 100
        benchmarks = self.run_benchmarks(start=start, end=end, starting_cash=starting_cash)
        return BacktestResult(
            start=start,
            end=end,
            starting_cash=starting_cash,
            ending_equity=ending_equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            per_symbol_pnl=per_symbol_pnl,
            benchmarks=benchmarks,
        )


def rank_adaptive_candidates(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    fast_lookback: int,
    slow_lookback: int,
    sector_lookback: int,
    min_sector_return: float,
    min_symbol_return: float,
    symbol_sma_filter_days: int = 0,
    use_sma_alignment: bool = False,
    use_relative_strength: bool = False,
    use_trend_consistency: bool = False,
    use_momentum_acceleration: bool = False,
    momentum_acceleration_weight: float = 6.0,
    slow_momentum_weight: float = 0.9,
    fast_momentum_weight: float = 1.8,
    short_momentum_weight: float = 0.6,
    volume_weight: float = 2.0,
    sector_weight_score: float = 0.5,
    high_proximity_weight: float = 15.0,
    sma_alignment_weight: float = 12.0,
    trend_consistency_weight: float = 8.0,
    gap_by_symbol: dict[str, float] | None = None,
    gap_boost_weight: float = 0.0,
    gap_boost_min_pct: float = 2.0,
) -> list[AdaptiveCandidate]:
    sector_return = sector_return_pct(bars_by_symbol, current_date, sector_lookback)
    if sector_return < min_sector_return:
        return []

    min_bars_needed = max(fast_lookback, slow_lookback, symbol_sma_filter_days, 50, 20)

    # Pre-compute sector average returns for relative strength scoring
    sector_avg_fast = 0.0
    sector_avg_slow = 0.0
    if use_relative_strength:
        fast_rets: list[float] = []
        slow_rets: list[float] = []
        for bars in bars_by_symbol.values():
            prior_rs = [b for b in bars if b.timestamp.date() < current_date]
            if len(prior_rs) <= min_bars_needed:
                continue
            if prior_rs[-fast_lookback - 1].close > 0:
                fast_rets.append(((prior_rs[-1].close / prior_rs[-fast_lookback - 1].close) - 1) * 100)
            if prior_rs[-slow_lookback - 1].close > 0:
                slow_rets.append(((prior_rs[-1].close / prior_rs[-slow_lookback - 1].close) - 1) * 100)
        sector_avg_fast = sum(fast_rets) / len(fast_rets) if fast_rets else 0.0
        sector_avg_slow = sum(slow_rets) / len(slow_rets) if slow_rets else 0.0

    candidates: list[AdaptiveCandidate] = []
    for symbol, bars in bars_by_symbol.items():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) <= min_bars_needed:
            continue
        latest = prior[-1]
        fast_base = prior[-fast_lookback - 1]
        slow_base = prior[-slow_lookback - 1]
        if fast_base.close <= 0 or slow_base.close <= 0:
            continue
        fast_return = ((latest.close / fast_base.close) - 1) * 100
        slow_return = ((latest.close / slow_base.close) - 1) * 100
        if fast_return < min_symbol_return or slow_return < min_symbol_return:
            continue

        # Hard SMA gate: discard if price is below the N-day SMA
        if symbol_sma_filter_days > 0:
            sma_gate = sum(b.close for b in prior[-symbol_sma_filter_days:]) / symbol_sma_filter_days
            if latest.close < sma_gate:
                continue

        avg_volume = sum(bar.volume for bar in prior[-20:]) / 20
        volume_ratio = latest.volume / avg_volume if avg_volume > 0 else 1.0

        # Short-term momentum boost (10d) supplements the 21d fast signal
        short_return = 0.0
        if len(prior) >= 12 and prior[-11].close > 0:
            short_return = ((latest.close / prior[-11].close) - 1) * 100

        # 52-week high proximity: stocks near/at new highs continue to outperform
        lookback_high = max(b.close for b in prior[-min(252, len(prior)):])
        high_proximity = latest.close / lookback_high if lookback_high > 0 else 1.0

        score = (
            slow_return * slow_momentum_weight
            + fast_return * fast_momentum_weight
            + short_return * short_momentum_weight
            + min(volume_ratio, 3.0) * volume_weight
            + sector_return * sector_weight_score
            + (high_proximity - 0.8) * high_proximity_weight
        )

        # Intraday gap boost: reward stocks with a large gap vs prior close (pre-market move)
        if gap_by_symbol and gap_boost_weight > 0:
            gap_pct = gap_by_symbol.get(symbol, 0.0)
            if gap_pct >= gap_boost_min_pct:
                score += gap_pct * gap_boost_weight

        sma_alignment = 0.0
        if use_sma_alignment:
            sma20 = sum(b.close for b in prior[-20:]) / 20
            sma50 = sum(b.close for b in prior[-50:]) / 50
            sma_alignment = (float(latest.close > sma20) + float(sma20 > sma50)) / 2
            score += sma_alignment * sma_alignment_weight

        # Relative strength: outperformance vs equal-weight sector average
        relative_strength = 0.0
        if use_relative_strength:
            rs_fast = fast_return - sector_avg_fast
            rs_slow = slow_return - sector_avg_slow
            relative_strength = rs_fast * 1.5 + rs_slow * 0.8
            score += relative_strength

        # Trend consistency: fraction of up-close days in fast window rewards smooth uptrends
        trend_consistency = 0.0
        if use_trend_consistency and len(prior) >= fast_lookback + 1:
            window = prior[-(fast_lookback + 1):]
            up_days = sum(1 for j in range(1, len(window)) if window[j].close > window[j - 1].close)
            trend_consistency = up_days / fast_lookback
            score += trend_consistency * trend_consistency_weight

        # Momentum acceleration: fast period gaining more per day than slow period
        # accel_ratio > 1 means the stock is speeding up (early breakout); < 1 = fading
        momentum_acceleration = 0.0
        if use_momentum_acceleration and slow_return > 0 and slow_lookback > 0:
            expected_fast = slow_return * fast_lookback / slow_lookback
            accel_ratio = fast_return / max(expected_fast, 0.1)
            momentum_acceleration = accel_ratio - 1.0
            score += max(-5.0, min(12.0, momentum_acceleration * momentum_acceleration_weight))

        candidates.append(
            AdaptiveCandidate(
                symbol=symbol,
                score=score,
                fast_return_pct=fast_return,
                slow_return_pct=slow_return,
                volume_ratio=volume_ratio,
                sma_alignment=sma_alignment,
                relative_strength=relative_strength,
                trend_consistency=trend_consistency,
                momentum_acceleration=momentum_acceleration,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def detect_rebound_context(
    bars_by_symbol: dict[str, list[DailyBar]],
    risk_bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    settings: dict[str, Any],
) -> ReboundContext:
    if not bool(settings.get("rebound_mode_enabled", False)):
        return ReboundContext(False, 0.0, 0.0, 0.0, 0.0, "rebound disabled")

    drawdown_lookback = int(settings.get("sector_drawdown_lookback_days", 63))
    fast_lookback = int(settings.get("rebound_fast_lookback_days", 5))
    breadth_sma_days = int(settings.get("rebound_breadth_sma_days", 5))
    trigger_drawdown = float(settings.get("rebound_trigger_drawdown_pct", -15.0))
    min_sector_fast = float(settings.get("rebound_min_sector_fast_return_pct", 3.0))
    min_market_fast = float(settings.get("rebound_min_market_fast_return_pct", 0.5))
    min_breadth = float(settings.get("rebound_min_breadth_pct", 0.45))

    drawdown = sector_drawdown_pct(bars_by_symbol, current_date, drawdown_lookback)
    sector_fast = sector_return_pct(bars_by_symbol, current_date, fast_lookback)
    market_fast = sector_return_pct(risk_bars_by_symbol, current_date, fast_lookback) if risk_bars_by_symbol else sector_fast
    breadth = sector_breadth_pct(bars_by_symbol, current_date, breadth_sma_days)

    failures: list[str] = []
    if drawdown > trigger_drawdown:
        failures.append(f"drawdown {drawdown:.2f}% > trigger {trigger_drawdown:.2f}%")
    if sector_fast < min_sector_fast:
        failures.append(f"sector fast {sector_fast:.2f}% < {min_sector_fast:.2f}%")
    if market_fast < min_market_fast:
        failures.append(f"market fast {market_fast:.2f}% < {min_market_fast:.2f}%")
    if breadth < min_breadth:
        failures.append(f"short breadth {breadth:.0%} < {min_breadth:.0%}")

    active = not failures
    reason = (
        f"rebound active: dd={drawdown:.2f}% sector_fast={sector_fast:.2f}% "
        f"market_fast={market_fast:.2f}% breadth={breadth:.0%}"
        if active
        else "rebound inactive: " + "; ".join(failures)
    )
    return ReboundContext(
        active=active,
        sector_drawdown_pct=drawdown,
        sector_fast_return_pct=sector_fast,
        market_fast_return_pct=market_fast,
        short_breadth_pct=breadth,
        reason=reason,
    )


def rank_rebound_candidates(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    settings: dict[str, Any],
) -> list[AdaptiveCandidate]:
    lookback = int(settings.get("rebound_symbol_lookback_days", 3))
    min_return = float(settings.get("rebound_min_symbol_return_pct", 2.0))
    volume_weight = float(settings.get("rebound_volume_weight", 2.0))
    drawdown_weight = float(settings.get("rebound_drawdown_weight", 0.25))
    min_bars_needed = max(lookback + 1, 21, 63)

    candidates: list[AdaptiveCandidate] = []
    for symbol, bars in bars_by_symbol.items():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) <= min_bars_needed:
            continue
        latest = prior[-1]
        base = prior[-lookback - 1]
        if base.close <= 0:
            continue
        rebound_return = ((latest.close / base.close) - 1) * 100
        if rebound_return < min_return:
            continue

        one_day_return = 0.0
        if prior[-2].close > 0:
            one_day_return = ((latest.close / prior[-2].close) - 1) * 100
        five_day_return = 0.0
        if prior[-6].close > 0:
            five_day_return = ((latest.close / prior[-6].close) - 1) * 100
        avg_volume = sum(bar.volume for bar in prior[-20:]) / 20
        volume_ratio = latest.volume / avg_volume if avg_volume > 0 else 1.0
        lookback_high = max(bar.close for bar in prior[-63:])
        drawdown_from_high = ((latest.close / lookback_high) - 1) * 100 if lookback_high > 0 else 0.0

        score = (
            rebound_return * 2.0
            + one_day_return
            + five_day_return * 0.6
            + min(volume_ratio, 4.0) * volume_weight
            + abs(min(drawdown_from_high, 0.0)) * drawdown_weight
        )
        candidates.append(
            AdaptiveCandidate(
                symbol=symbol,
                score=score,
                fast_return_pct=rebound_return,
                slow_return_pct=five_day_return,
                volume_ratio=volume_ratio,
            )
        )

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def sector_breadth_pct(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    sma_days: int,
) -> float:
    """Fraction of watchlist symbols trading above their N-day SMA."""
    above = 0
    total = 0
    for bars in bars_by_symbol.values():
        prior = [b for b in bars if b.timestamp.date() < current_date]
        if len(prior) < sma_days:
            continue
        sma = sum(b.close for b in prior[-sma_days:]) / sma_days
        total += 1
        if prior[-1].close > sma:
            above += 1
    return above / total if total > 0 else 1.0


def compute_atr_weights(
    candidates: list[AdaptiveCandidate],
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    atr_days: int,
) -> list[float]:
    """Inverse-ATR weights: lower-volatility symbols receive proportionally more capital."""
    inv_atrs: list[float] = []
    for candidate in candidates:
        bars = bars_by_symbol.get(candidate.symbol, [])
        prior = [b for b in bars if b.timestamp.date() < current_date]
        if len(prior) < atr_days + 1:
            inv_atrs.append(1.0)
            continue
        window = prior[-(atr_days + 1):]
        trs = [
            max(
                window[i].high - window[i].low,
                abs(window[i].high - window[i - 1].close),
                abs(window[i].low - window[i - 1].close),
            )
            for i in range(1, len(window))
        ]
        atr = sum(trs) / atr_days
        atr_pct = atr / window[-1].close if window[-1].close > 0 else 1.0
        inv_atrs.append(1.0 / max(atr_pct, 0.001))
    total = sum(inv_atrs)
    return [v / total for v in inv_atrs] if total > 0 else [1.0 / len(candidates)] * len(candidates)


def sector_return_pct(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    lookback_days: int,
) -> float:
    returns: list[float] = []
    for bars in bars_by_symbol.values():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) <= lookback_days:
            continue
        latest = prior[-1]
        base = prior[-lookback_days - 1]
        if base.close > 0:
            returns.append(((latest.close / base.close) - 1) * 100)
    return sum(returns) / len(returns) if returns else 0.0


def should_retain_winner(
    symbol: str,
    position: BacktestPosition,
    current_price: float,
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    fast_lookback: int,
    slow_lookback: int,
    min_profit_pct: float,
    min_fast_momentum_pct: float,
    min_slow_momentum_pct: float,
) -> bool:
    if position.qty <= 0 or position.avg_entry <= 0:
        return False
    profit_pct = ((current_price / position.avg_entry) - 1) * 100
    if profit_pct < min_profit_pct:
        return False

    prior = [bar for bar in bars_by_symbol.get(symbol, []) if bar.timestamp.date() < current_date]
    if len(prior) <= max(fast_lookback, slow_lookback):
        return False
    latest = prior[-1]
    fast_base = prior[-fast_lookback - 1]
    slow_base = prior[-slow_lookback - 1]
    if fast_base.close <= 0 or slow_base.close <= 0:
        return False
    fast_return = ((latest.close / fast_base.close) - 1) * 100
    slow_return = ((latest.close / slow_base.close) - 1) * 100
    return fast_return >= min_fast_momentum_pct and slow_return >= min_slow_momentum_pct


def apply_buzz_earnings_overlay(
    candidates: list[AdaptiveCandidate],
    current_date: date,
    positions: dict[str, BacktestPosition],
    today_bars: dict[str, DailyBar],
    news_by_symbol: dict[str, list[tuple[date, str]]],
    earnings_by_symbol: dict[str, set[date]],
    settings: dict[str, Any],
) -> list[AdaptiveCandidate]:
    if not bool(settings.get("enabled", False)):
        return candidates

    adjusted: list[AdaptiveCandidate] = []
    for candidate in candidates:
        bar = today_bars.get(candidate.symbol)
        if not bar:
            continue
        if is_earnings_blocked(
            symbol=candidate.symbol,
            current_date=current_date,
            position=positions[candidate.symbol],
            current_price=bar.close,
            earnings_by_symbol=earnings_by_symbol,
            settings=settings,
        ):
            continue

        buzz = buzz_snapshot(
            symbol=candidate.symbol,
            current_date=current_date,
            news_by_symbol=news_by_symbol,
            settings=settings,
        )
        if buzz.score <= float(settings.get("negative_score_block_threshold", -3.0)):
            continue
        score_boost = max(
            -float(settings.get("max_score_boost", 20.0)),
            min(
                float(settings.get("max_score_boost", 20.0)),
                buzz.score * float(settings.get("score_weight", 2.0)),
            ),
        )
        adjusted.append(
            AdaptiveCandidate(
                symbol=candidate.symbol,
                score=candidate.score + score_boost,
                fast_return_pct=candidate.fast_return_pct,
                slow_return_pct=candidate.slow_return_pct,
                volume_ratio=candidate.volume_ratio,
            )
        )

    return sorted(adjusted, key=lambda item: item.score, reverse=True)


def buzz_snapshot(
    symbol: str,
    current_date: date,
    news_by_symbol: dict[str, list[tuple[date, str]]],
    settings: dict[str, Any],
) -> BuzzSnapshot:
    lookback_days = int(settings.get("news_lookback_days", 7))
    positive_keywords = [str(item).lower() for item in settings.get("positive_keywords", [])]
    negative_keywords = [str(item).lower() for item in settings.get("negative_keywords", [])]
    start_date = current_date - timedelta(days=lookback_days)

    article_count = 0
    positive_hits = 0
    negative_hits = 0
    for news_date, text in news_by_symbol.get(symbol, []):
        if not start_date <= news_date < current_date:
            continue
        article_count += 1
        positive_hits += sum(1 for keyword in positive_keywords if keyword and keyword in text)
        negative_hits += sum(1 for keyword in negative_keywords if keyword and keyword in text)

    score = (
        min(article_count, int(settings.get("article_count_cap", 5))) * float(settings.get("article_weight", 0.4))
        + positive_hits * float(settings.get("positive_keyword_weight", 1.0))
        - negative_hits * float(settings.get("negative_keyword_weight", 2.0))
    )
    return BuzzSnapshot(
        article_count=article_count,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        score=score,
    )


def is_earnings_blocked(
    symbol: str,
    current_date: date,
    position: BacktestPosition,
    current_price: float,
    earnings_by_symbol: dict[str, set[date]],
    settings: dict[str, Any],
) -> bool:
    if not bool(settings.get("block_new_entries_near_earnings", True)):
        return False

    before_days = int(settings.get("earnings_avoid_days_before", 2))
    after_days = int(settings.get("earnings_avoid_days_after", 1))
    nearby = [
        earnings_date
        for earnings_date in earnings_by_symbol.get(symbol, set())
        if -after_days <= (earnings_date - current_date).days <= before_days
    ]
    if not nearby:
        return False
    if position.qty <= 0 or position.avg_entry <= 0:
        return True
    profit_cushion = ((current_price / position.avg_entry) - 1) * 100
    return profit_cushion < float(settings.get("allow_earnings_if_profit_cushion_pct", 5.0))


def fetch_news_by_symbol(
    api_key: str,
    secret_key: str,
    symbols: list[str],
    start: date,
    end: date,
    settings: dict[str, Any],
) -> dict[str, list[tuple[date, str]]]:
    if not bool(settings.get("enabled", False)) or not bool(settings.get("news_enabled", True)):
        return {}

    result: dict[str, list[tuple[date, str]]] = {symbol: [] for symbol in symbols}
    client = NewsClient(api_key=api_key, secret_key=secret_key)
    chunk_days = int(settings.get("news_fetch_chunk_days", 30))
    limit = int(settings.get("news_limit_per_request", 50))
    try:
        for symbol in symbols:
            chunk_start = start
            while chunk_start <= end:
                chunk_end = min(end, chunk_start + timedelta(days=chunk_days))
                request = NewsRequest(
                    symbols=symbol,
                    start=datetime.combine(chunk_start, datetime.min.time(), tzinfo=UTC),
                    end=datetime.combine(chunk_end, datetime.max.time(), tzinfo=UTC),
                    limit=limit,
                    include_content=True,
                )
                response = client.get_news(request)
                for article in extract_news_articles(response):
                    article_date = news_article_date(article)
                    if article_date is None:
                        continue
                    result.setdefault(symbol, []).append((article_date, news_article_text(article)))
                chunk_start = chunk_end + timedelta(days=1)
    except Exception as error:
        if bool(settings.get("fail_open", True)):
            print(f"Buzz/news overlay unavailable; continuing without news scores ({type(error).__name__})")
            return {}
        raise

    return result


def extract_news_articles(response: Any) -> list[Any]:
    raw_articles = getattr(response, "data", getattr(response, "news", response))
    if isinstance(raw_articles, dict):
        articles: list[Any] = []
        for value in raw_articles.values():
            if isinstance(value, list):
                articles.extend(value)
            else:
                articles.append(value)
        return articles
    if raw_articles is None:
        return []
    try:
        return list(raw_articles)
    except TypeError:
        return [raw_articles]


def news_article_date(article: Any) -> date | None:
    value = getattr(article, "created_at", None)
    if isinstance(article, dict):
        value = article.get("created_at", value)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def news_article_text(article: Any) -> str:
    fields = [
        getattr(article, "headline", ""),
        getattr(article, "summary", ""),
        getattr(article, "content", ""),
    ]
    if isinstance(article, dict):
        fields.extend([article.get("headline", ""), article.get("summary", ""), article.get("content", "")])
    return " ".join(str(value).lower() for value in fields if value)


def load_earnings_calendar(path_value: Any) -> dict[str, set[date]]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}

    result: dict[str, set[date]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            symbol = str(row.get("symbol", "")).strip().upper()
            raw_date = row.get("date") or row.get("earnings_date")
            if not symbol or not raw_date:
                continue
            try:
                earnings_date = date.fromisoformat(str(raw_date).strip())
            except ValueError:
                continue
            result.setdefault(symbol, set()).add(earnings_date)
    return result


def sector_drawdown_pct(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    lookback_days: int,
) -> float:
    drawdowns: list[float] = []
    for bars in bars_by_symbol.values():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) < lookback_days:
            continue
        window = prior[-lookback_days:]
        peak = max(bar.close for bar in window)
        latest = window[-1].close
        if peak > 0:
            drawdowns.append(((latest - peak) / peak) * 100)
    return sum(drawdowns) / len(drawdowns) if drawdowns else 0.0


def market_filter_allows_entries(
    bars_by_symbol: dict[str, list[DailyBar]],
    current_date: date,
    momentum_lookback: int,
    sma_days: int,
    min_momentum_pct: float,
    require_above_sma: bool,
) -> bool:
    if not bars_by_symbol:
        return True

    usable_symbols = 0
    for bars in bars_by_symbol.values():
        prior = [bar for bar in bars if bar.timestamp.date() < current_date]
        if len(prior) <= max(momentum_lookback, sma_days):
            continue
        usable_symbols += 1
        latest = prior[-1]
        momentum_base = prior[-momentum_lookback - 1]
        if momentum_base.close <= 0:
            return False
        momentum_pct = ((latest.close / momentum_base.close) - 1) * 100
        if momentum_pct < min_momentum_pct:
            return False
        if require_above_sma:
            sma = sum(bar.close for bar in prior[-sma_days:]) / sma_days
            if latest.close < sma:
                return False

    return usable_symbols > 0


def index_daily_bars_by_date(bars_by_symbol: dict[str, list[DailyBar]]) -> dict[date, dict[str, DailyBar]]:
    bars_by_date: dict[date, dict[str, DailyBar]] = {}
    for symbol, bars in bars_by_symbol.items():
        for bar in bars:
            bars_by_date.setdefault(bar.timestamp.date(), {})[symbol] = bar
    return bars_by_date


def sell_position(
    trades: list[BacktestTrade],
    per_symbol_pnl: dict[str, float],
    cash: float,
    position: BacktestPosition,
    symbol: str,
    timestamp: datetime,
    price: float,
    reason: str,
) -> float:
    qty = position.qty
    notional = qty * price
    per_symbol_pnl[symbol] = per_symbol_pnl.get(symbol, 0.0) + (price - position.avg_entry) * qty
    cash_after = cash + notional
    trades.append(
        BacktestTrade(
            timestamp=timestamp,
            symbol=symbol,
            action="sell",
            qty=qty,
            price=price,
            notional=notional,
            cash_after=cash_after,
            reason=reason,
        )
    )
    position.qty = 0.0
    position.avg_entry = 0.0
    position.peak_price = 0.0
    return cash_after


def fetch_live_prices(bot: SemiMomentumBot, symbols: list[str]) -> dict[str, float]:
    snapshots = bot.data.get_stock_snapshot(
        StockSnapshotRequest(symbol_or_symbols=symbols, feed=bot.feed)
    )
    prices: dict[str, float] = {}
    for symbol in symbols:
        snapshot = snapshots.get(symbol)
        latest_trade = getattr(snapshot, "latest_trade", None) if snapshot else None
        if latest_trade is None:
            continue
        prices[symbol] = float(latest_trade.price)
    return prices


def _compute_gap_by_symbol(
    latest_prices: dict[str, float],
    bars_by_symbol: dict[str, list[DailyBar]],
) -> dict[str, float]:
    """Return intraday gap pct (vs prior close) for each symbol with live price data."""
    gaps: dict[str, float] = {}
    for symbol, price in latest_prices.items():
        hist = bars_by_symbol.get(symbol, [])
        if not hist:
            continue
        prev_close = hist[-1].close
        if prev_close > 0:
            gaps[symbol] = ((price / prev_close) - 1) * 100
    return gaps


def fetch_extended_hours_gaps(bot: SemiMomentumBot, symbols: list[str]) -> dict[str, float]:
    """Return extended-hours gap pct vs the most recent regular-session close.

    Alpaca daily_bar starts at midnight ET and its .close reflects the current/last
    trade price (not a locked 4pm print).  The correct reference is therefore:

      Pre-market  (before 16:00 ET): previous_daily_bar.close  ← yesterday's 4pm close
      After-hours (after  16:00 ET): daily_bar.close           ← today's 4pm close (locked)

    gap = (latest_trade.price / reference_close) - 1
    """
    now_et = datetime.now(MARKET_TZ)
    use_prev = now_et.hour < 16  # before 4pm ET → pre-market; use yesterday's close

    snapshots = bot.data.get_stock_snapshot(
        StockSnapshotRequest(symbol_or_symbols=symbols, feed=bot.feed)
    )
    gaps: dict[str, float] = {}
    for symbol in symbols:
        snapshot = snapshots.get(symbol)
        if not snapshot:
            continue
        latest = getattr(snapshot, "latest_trade", None)
        if not latest:
            continue
        price = float(latest.price)

        day_bar = getattr(snapshot, "daily_bar", None)
        prev_bar = getattr(snapshot, "previous_daily_bar", None)

        if use_prev:
            # Pre-market: yesterday's 4pm close is the reference
            ref_bar = prev_bar
            fallback = day_bar
        else:
            # After-hours: today's locked 4pm close is the reference
            ref_bar = day_bar
            fallback = prev_bar

        reference = None
        if ref_bar and getattr(ref_bar, "close", None):
            reference = float(ref_bar.close)
        elif fallback and getattr(fallback, "close", None):
            reference = float(fallback.close)

        if reference and reference > 0:
            gaps[symbol] = ((price / reference) - 1) * 100
    return gaps


# backward-compat alias for old callers
def fetch_afterhours_gaps(bot: SemiMomentumBot, symbols: list[str]) -> dict[str, float]:
    return fetch_extended_hours_gaps(bot, symbols)


def live_overlay_bars(
    bars_by_symbol: dict[str, list[DailyBar]],
    prices: dict[str, float],
    current_date: date,
) -> dict[str, DailyBar]:
    bars: dict[str, DailyBar] = {}
    for symbol, price in prices.items():
        historical_bars = bars_by_symbol.get(symbol, [])
        previous = historical_bars[-1] if historical_bars else None
        volume = previous.volume if previous else 0.0
        open_price = previous.close if previous else price
        bars[symbol] = DailyBar(
            symbol=symbol,
            timestamp=datetime.combine(current_date, datetime.min.time(), tzinfo=UTC),
            open=open_price,
            high=max(open_price, price),
            low=min(open_price, price),
            close=price,
            volume=volume,
        )
    return bars


def backtest_positions_from_live(
    watchlist: list[str],
    positions: dict[str, Any],
) -> dict[str, BacktestPosition]:
    mapped = {symbol: BacktestPosition() for symbol in watchlist}
    for symbol in watchlist:
        position = positions.get(symbol)
        if not position:
            continue
        qty = float(getattr(position, "qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        avg_entry = float(getattr(position, "avg_entry_price", 0.0) or 0.0)
        market_value = abs(float(getattr(position, "market_value", 0.0) or 0.0))
        current_price = market_value / qty if qty > 0 and market_value > 0 else avg_entry
        mapped[symbol] = BacktestPosition(
            qty=qty,
            avg_entry=avg_entry,
            peak_price=max(avg_entry, current_price),
        )
    return mapped


def submit_adaptive_decisions(
    bot: SemiMomentumBot,
    decisions: list[Decision],
    dry_run: bool,
    max_orders: int,
    extended_hours: bool = False,
) -> list[Decision]:
    submitted: list[Decision] = []
    for decision in decisions:
        if len(submitted) >= max_orders:
            break
        if decision.action == "hold":
            print(f"ADAPTIVE HOLD {format_decision(decision)}")
            bot.log_decision(decision, event="adaptive_hold")
            continue
        if dry_run:
            print(f"ADAPTIVE DRY RUN {format_decision(decision)}")
            bot.log_decision(decision, event="adaptive_dry_run_order")
        else:
            bot.submit_order(decision, extended_hours=extended_hours)
            bot.log_decision(decision, event="adaptive_submitted_order")
        submitted.append(decision)
    return submitted


def print_adaptive_allocator_result(result: BacktestResult) -> None:
    print("Adaptive semi portfolio")
    print_backtest_result(result)


def write_adaptive_allocator_trades(path: str, trades: list[BacktestTrade]) -> None:
    write_trades_csv(path, trades)
