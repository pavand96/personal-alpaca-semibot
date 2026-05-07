from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from semibot.events import append_event


@dataclass(frozen=True)
class MarketView:
    symbol: str
    price: float
    previous_close: float
    change_pct: float
    held_qty: float
    held_market_value: float


@dataclass(frozen=True)
class Decision:
    symbol: str
    action: str
    reason: str
    notional: float = 0.0
    qty: float = 0.0


class SemiMomentumBot:
    def __init__(self, config: dict[str, Any], api_key: str, secret_key: str) -> None:
        self.config = config
        self.trading = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=bool(config["alpaca"]["paper"]),
        )
        self.data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.feed = parse_data_feed(config["alpaca"].get("data_feed", "iex"))

    def monitor(self) -> list[MarketView]:
        views = self.get_market_views()
        for view in sorted(views, key=lambda item: item.change_pct, reverse=True):
            print(
                f"{view.symbol:5} price=${view.price:9.2f} "
                f"prev_close=${view.previous_close:9.2f} "
                f"change={view.change_pct:7.2f}% "
                f"held_qty={view.held_qty:g}"
            )
            self.log_view(view)
        return views

    def trade_once(self, execute: bool = False) -> list[Decision]:
        self.assert_account_can_trade()

        if self.config["risk"]["require_market_open"] and not self.trading.get_clock().is_open:
            print("Market is closed. No orders submitted.")
            return []

        dry_run = bool(self.config["risk"]["dry_run"]) or not execute
        max_orders = int(self.config["risk"]["max_orders_per_run"])
        if self.daily_loss_kill_switch_triggered():
            positions = self.get_positions()
            if self.config["risk"].get("flatten_on_daily_loss", True):
                decisions = [
                    Decision(position.symbol, "sell", "daily loss kill switch", qty=float(position.qty))
                    for position in positions.values()
                    if float(position.qty) > 0
                ]
            else:
                print("Daily loss kill switch active. No orders submitted.")
                return []
        else:
            decisions = self.decide(self.get_market_views())

        submitted: list[Decision] = []
        for decision in decisions:
            if len(submitted) >= max_orders:
                break
            if decision.action == "hold":
                continue

            if dry_run:
                print(f"DRY RUN {format_decision(decision)}")
                self.log_decision(decision, event="dry_run_order")
            else:
                self.submit_order(decision)
                self.log_decision(decision, event="submitted_order")
            submitted.append(decision)

        if not submitted:
            print("No trade signals crossed the configured thresholds.")
        return submitted

    def decide(self, views: list[MarketView]) -> list[Decision]:
        strategy = self.config["strategy"]
        buy_threshold = float(strategy["buy_threshold_pct"])
        sell_threshold = float(strategy["sell_threshold_pct"])
        per_trade_notional = float(strategy["per_trade_notional"])
        max_position_notional = float(strategy["max_position_notional"])
        max_buys = int(strategy["max_symbols_to_buy_per_run"])
        min_price = float(strategy["min_price"])
        max_price = float(strategy["max_price"])

        decisions: list[Decision] = []
        buys_used = 0

        for view in sorted(views, key=lambda item: item.change_pct, reverse=True):
            if view.price < min_price or view.price > max_price:
                decisions.append(Decision(view.symbol, "hold", "outside configured price bounds"))
                continue

            if view.held_qty > 0 and view.change_pct <= sell_threshold:
                decisions.append(
                    Decision(
                        symbol=view.symbol,
                        action="sell",
                        qty=view.held_qty,
                        reason=f"change {view.change_pct:.2f}% <= sell threshold {sell_threshold:.2f}%",
                    )
                )
                continue

            can_add = view.held_market_value + per_trade_notional <= max_position_notional
            if (
                view.change_pct >= buy_threshold
                and can_add
                and buys_used < max_buys
            ):
                buys_used += 1
                decisions.append(
                    Decision(
                        symbol=view.symbol,
                        action="buy",
                        notional=per_trade_notional,
                        reason=f"change {view.change_pct:.2f}% >= buy threshold {buy_threshold:.2f}%",
                    )
                )
                continue

            if view.change_pct >= buy_threshold and not can_add:
                decisions.append(Decision(view.symbol, "hold", "max position notional reached"))
            else:
                decisions.append(Decision(view.symbol, "hold", "no threshold crossed"))

        return decisions

    def get_market_views(self) -> list[MarketView]:
        symbols = self.config["watchlist"]
        snapshots = self.data.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbols, feed=self.feed)
        )
        positions = self.get_positions()

        views: list[MarketView] = []
        for symbol in symbols:
            snapshot = snapshots.get(symbol)
            if not snapshot or not snapshot.latest_trade or not snapshot.previous_daily_bar:
                continue

            price = float(snapshot.latest_trade.price)
            previous_close = float(snapshot.previous_daily_bar.close)
            if previous_close <= 0:
                continue

            position = positions.get(symbol)
            held_qty = float(position.qty) if position else 0.0
            held_market_value = abs(float(position.market_value)) if position else 0.0
            change_pct = ((price - previous_close) / previous_close) * 100

            views.append(
                MarketView(
                    symbol=symbol,
                    price=price,
                    previous_close=previous_close,
                    change_pct=change_pct,
                    held_qty=held_qty,
                    held_market_value=held_market_value,
                )
            )
        return views

    def get_positions(self) -> dict[str, Any]:
        return {position.symbol: position for position in self.trading.get_all_positions()}

    def submit_order(self, decision: Decision) -> None:
        side = OrderSide.BUY if decision.action == "buy" else OrderSide.SELL
        order_type = str(self.config.get("orders", {}).get("type", "market")).lower()
        request_kwargs: dict[str, Any] = {
            "symbol": decision.symbol,
            "side": side,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": f"semibot-{decision.symbol}-{uuid4().hex[:12]}",
        }

        if order_type == "limit":
            reference_price = self.get_latest_price(decision.symbol)
            offset_bps = float(self.config.get("orders", {}).get("limit_price_offset_bps", 10.0))
            limit_price = limit_price_for_side(reference_price, decision.action, offset_bps)
            request_kwargs["limit_price"] = limit_price
            if decision.action == "buy":
                request_kwargs["qty"] = round(decision.notional / limit_price, 6)
            else:
                request_kwargs["qty"] = decision.qty
            order = LimitOrderRequest(**request_kwargs)
        else:
            if decision.action == "buy":
                request_kwargs["notional"] = round(decision.notional, 2)
            else:
                request_kwargs["qty"] = decision.qty
            order = MarketOrderRequest(**request_kwargs)

        result = self.trading.submit_order(order_data=order)
        print(f"Submitted {order_type.upper()} {decision.action.upper()} {decision.symbol}: order_id={result.id}")

    def get_latest_price(self, symbol: str) -> float:
        snapshots = self.data.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=[symbol], feed=self.feed)
        )
        snapshot = snapshots.get(symbol)
        if not snapshot or not snapshot.latest_trade:
            raise RuntimeError(f"No latest trade available for {symbol}")
        return float(snapshot.latest_trade.price)

    def assert_account_can_trade(self) -> None:
        account = self.trading.get_account()
        if getattr(account, "trading_blocked", False):
            raise RuntimeError("Alpaca account is trading blocked")
        if getattr(account, "account_blocked", False):
            raise RuntimeError("Alpaca account is account blocked")

    def daily_loss_kill_switch_triggered(self) -> bool:
        max_daily_loss_pct = float(self.config["risk"].get("max_daily_loss_pct", 0.0))
        if max_daily_loss_pct <= 0:
            return False
        account = self.trading.get_account()
        daily_return_pct = account_daily_return_pct(account)
        if daily_return_pct <= -max_daily_loss_pct:
            print(
                f"Daily loss kill switch active: account daily return "
                f"{daily_return_pct:.2f}% <= -{max_daily_loss_pct:.2f}%"
            )
            return True
        return False

    def log_view(self, view: MarketView) -> None:
        append_event(
            self.config["runtime"]["log_file"],
            {
                "event": "market_view",
                "symbol": view.symbol,
                "price": round(view.price, 4),
                "previous_close": round(view.previous_close, 4),
                "change_pct": round(view.change_pct, 4),
                "quantity": view.held_qty,
            },
        )

    def log_decision(self, decision: Decision, event: str) -> None:
        append_event(
            self.config["runtime"]["log_file"],
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "event": event,
                "symbol": decision.symbol,
                "action": decision.action,
                "quantity": decision.qty,
                "notional": decision.notional,
                "reason": decision.reason,
            },
        )


def parse_data_feed(value: str) -> DataFeed:
    normalized = str(value).strip().upper()
    try:
        return getattr(DataFeed, normalized)
    except AttributeError as error:
        valid = ", ".join(item.name.lower() for item in DataFeed)
        raise ValueError(f"Unsupported data_feed '{value}'. Choose one of: {valid}") from error


def format_decision(decision: Decision) -> str:
    if decision.action == "buy":
        size = f"${decision.notional:.2f}"
    else:
        size = f"{decision.qty:g} shares"
    return f"{decision.action.upper()} {decision.symbol} {size}: {decision.reason}"


def account_daily_return_pct(account: Any) -> float:
    equity = float(getattr(account, "equity", 0.0) or 0.0)
    last_equity = float(getattr(account, "last_equity", 0.0) or 0.0)
    if last_equity <= 0:
        return 0.0
    return ((equity / last_equity) - 1) * 100


def limit_price_for_side(reference_price: float, action: str, offset_bps: float) -> float:
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    multiplier = 1 + (offset_bps / 10000)
    if action == "sell":
        multiplier = 1 - (offset_bps / 10000)
    return round(reference_price * multiplier, 2)
