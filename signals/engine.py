"""Single causal feature engine shared by live-tail processing and replay."""

import math
import statistics
from copy import deepcopy

from streaming import ASSET_SERIES
from streaming.timeutil import parse_timestamp

from .config import SignalConfig
from .mathutil import mad, mean, median, safe_ratio, stddev, log_return
from .state import AssetState, ContractState


def canonical_probability(yes_bid, yes_ask, no_bid, no_ask):
    """Executable UP midpoint using direct and complementary YES prices."""
    if any(value is None for value in (yes_bid, yes_ask, no_bid, no_ask)):
        return None, None, None, None
    best_bid = max(yes_bid, 1.0 - no_ask)
    best_ask = min(yes_ask, 1.0 - no_bid)
    if not (0 <= best_bid <= best_ask <= 1):
        return None, None, best_bid, best_ask
    return (best_bid + best_ask) / 2, best_ask - best_bid, best_bid, best_ask


def _float(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class SignalEngine:
    def __init__(self, config=None):
        self.config = config or SignalConfig()
        self.assets = {asset: AssetState.create(asset, self.config) for asset in ASSET_SERIES}
        self.last_event_time = None
        self.event_count = 0

    def process(self, event):
        """Consume one raw event mapping and return a frozen feature snapshot."""
        timestamp = parse_timestamp(event["local_receive_timestamp"]).timestamp()
        if self.last_event_time is not None and timestamp < self.last_event_time:
            raise ValueError("raw events are not in causal order")
        self.last_event_time = timestamp
        self.event_count += 1
        asset = event["asset"]
        if asset not in self.assets:
            return None
        state = self.assets[asset]
        payload = event.get("raw_payload") or {}
        kind = event["event_type"]
        self._update_contract(state, event, timestamp)
        if kind == "ticker":
            self._ticker(state, payload, timestamp)
        elif kind == "underlying_price":
            self._underlying(state, payload, timestamp)
        elif kind == "orderbook_snapshot":
            self._book_snapshot(state, payload, event.get("source", ""), timestamp)
        elif kind == "orderbook_delta":
            self._book_delta(state, payload, timestamp)
        elif kind == "trade":
            self._trade(state, payload, timestamp)
        elif kind == "contract_reset":
            self._reset(state, payload, event, timestamp)
        elif kind == "market_lifecycle":
            self._lifecycle(state, payload)
        return self.snapshot(asset, timestamp, event.get("event_id"))

    def _update_contract(self, state, event, timestamp):
        ticker = event.get("market_ticker")
        if ticker and state.contract.ticker is None:
            state.contract.ticker = ticker
        for attr, key in (("open_time", "contract_open_time"),
                          ("close_time", "contract_close_time")):
            if event.get(key):
                setattr(state.contract, attr, parse_timestamp(event[key]).timestamp())
        if event.get("target") is not None:
            state.contract.target = float(event["target"])

    def _ticker(self, state, payload, timestamp):
        msg = payload.get("msg", payload)
        def dollars(name, cents_name=None):
            value = _float(msg.get(name))
            if value is None and cents_name:
                cents = _float(msg.get(cents_name))
                value = None if cents is None else cents / 100
            return value
        state.yes_bid = dollars("yes_bid_dollars", "yes_bid")
        state.yes_ask = dollars("yes_ask_dollars", "yes_ask")
        state.no_bid = dollars("no_bid_dollars", "no_bid")
        state.no_ask = dollars("no_ask_dollars", "no_ask")
        if state.no_bid is None and state.yes_ask is not None:
            state.no_bid = 1 - state.yes_ask
        if state.no_ask is None and state.yes_bid is not None:
            state.no_ask = 1 - state.yes_bid
        state.last_trade = dollars("price_dollars")
        if state.last_trade is None:
            state.last_trade = dollars("last_price_dollars", "last_price")
        state.volume = _float(msg.get("volume_fp", msg.get("volume")))
        state.open_interest = _float(msg.get("open_interest_fp", msg.get("open_interest")))
        state.quote_time = timestamp
        probability, _, _, _ = canonical_probability(
            state.yes_bid, state.yes_ask, state.no_bid, state.no_ask)
        if probability is not None:
            state.probability.append(timestamp, probability)
            state.contract.probability_values.append((timestamp, probability))
            for window, history in state.velocity_histories.items():
                velocity = state.probability.velocity(timestamp, window)
                if velocity is not None:
                    history.append(timestamp, velocity)

    def _underlying(self, state, payload, timestamp):
        msg = payload.get("msg", payload)
        price = _float(msg.get("usd_price"))
        if price is not None and price > 0:
            state.price.append(timestamp, price)
            state.contract.price_values.append((timestamp, price))

    def _book_snapshot(self, state, payload, source, timestamp):
        msg = payload.get("msg", payload)
        rest = payload.get("orderbook_fp") if isinstance(payload, dict) else None
        if rest is not None:
            yes = rest.get("yes_dollars", [])
            no = rest.get("no_dollars", [])
            state.bid_book = {float(price): float(qty) for price, qty in yes}
            state.ask_book = {1 - float(price): float(qty) for price, qty in no}
        else:
            yes = msg.get("yes_dollars_fp", [])
            no = msg.get("no_dollars_fp", [])
            state.bid_book = {float(price): float(qty) for price, qty in yes}
            # Phase 2 subscribes with use_yes_price=true, so NO book is YES ask scale.
            state.ask_book = {float(price): float(qty) for price, qty in no}
        state.book_time = timestamp

    def _book_delta(self, state, payload, timestamp):
        msg = payload.get("msg", payload)
        side, price, delta = msg.get("side"), _float(msg.get("price_dollars")), _float(msg.get("delta_fp"))
        if side not in ("yes", "no") or price is None or delta is None:
            return
        book = state.bid_book if side == "yes" else state.ask_book
        quantity = book.get(price, 0.0) + delta
        if quantity <= 0:
            book.pop(price, None)
        else:
            book[price] = quantity
        state.book_time = timestamp

    def _trade(self, state, payload, timestamp):
        msg = payload.get("msg", payload)
        count = _float(msg.get("count_fp"))
        yes_price = _float(msg.get("yes_price_dollars"))
        direction = msg.get("taker_outcome_side") or msg.get("taker_side")
        if direction not in ("yes", "no"):
            direction = None
        if count is not None:
            state.trades.append((timestamp, count, direction, yes_price))
            cutoff = timestamp - self.config.history_retention_seconds
            state.trades[:] = [trade for trade in state.trades if trade[0] >= cutoff]
        if yes_price is not None:
            state.last_trade = yes_price

    def _lifecycle(self, state, payload):
        msg = payload.get("msg", payload)
        result = msg.get("result") or msg.get("market_result")
        if result in ("yes", "no"):
            state.contract.result = result

    def _reset(self, state, payload, event, timestamp):
        prior = self._finalize_contract(state, payload)
        state.prior_window = prior
        state.contract = ContractState(
            ticker=payload.get("new_ticker") or event.get("market_ticker"),
            open_time=(parse_timestamp(event["contract_open_time"]).timestamp()
                       if event.get("contract_open_time") else timestamp),
            close_time=(parse_timestamp(event["contract_close_time"]).timestamp()
                        if event.get("contract_close_time") else None),
            target=_float(event.get("target")))
        state.probability = type(state.probability)(self.config.history_retention_seconds)
        state.velocity_histories = {
            window: type(history)(self.config.history_retention_seconds)
            for window, history in state.velocity_histories.items()}
        state.yes_bid = state.yes_ask = state.no_bid = state.no_ask = None
        state.last_trade = state.volume = state.open_interest = None
        state.quote_time = state.book_time = None
        state.bid_book.clear(); state.ask_book.clear(); state.trades.clear()
        state.checkpoints_emitted.clear()

    def _finalize_contract(self, state, reset_payload):
        prices = [value for _, value in state.contract.price_values]
        probabilities = [value for _, value in state.contract.probability_values]
        returns = [log_return(prices[i - 1], prices[i]) for i in range(1, len(prices))]
        returns = [value for value in returns if value is not None]
        prior_final = reset_payload.get("prior_contract_final_market_state") or {}
        result = reset_payload.get("previous_settlement") or prior_final.get("result") or state.contract.result
        prior_return = safe_ratio(prices[-1] - prices[0], prices[0]) if len(prices) >= 2 else None
        return {
            "prior_result": result,
            "prior_open_reference": prices[0] if prices else None,
            "prior_close_reference": prices[-1] if prices else None,
            "prior_return": prior_return,
            "prior_absolute_return": None if prior_return is None else abs(prior_return),
            "prior_final_probability": probabilities[-1] if probabilities else None,
            "prior_probability_extremity": (None if not probabilities else
                                             abs(probabilities[-1] - .5)),
            "prior_max_probability": max(probabilities) if probabilities else None,
            "prior_min_probability": min(probabilities) if probabilities else None,
            "prior_realized_volatility": (math.sqrt(sum(value * value for value in returns))
                                           if returns else None),
        }

    def snapshot(self, asset, timestamp, watermark):
        state = self.assets[asset]
        probability, spread, executable_bid, executable_ask = canonical_probability(
            state.yes_bid, state.yes_ask, state.no_bid, state.no_ask)
        quote_age = None if state.quote_time is None else timestamp - state.quote_time
        book_age = None if state.book_time is None else timestamp - state.book_time
        quote_fresh = quote_age is not None and quote_age <= self.config.quote_stale_seconds
        book_fresh = book_age is not None and book_age <= self.config.book_stale_seconds
        if not quote_fresh:
            probability, spread = None, None
        features = {
            "yes_bid": state.yes_bid, "yes_ask": state.yes_ask,
            "no_bid": state.no_bid, "no_ask": state.no_ask,
            "midpoint_up_probability": probability, "spread": spread,
            "last_trade": state.last_trade, "quote_age_seconds": quote_age,
            "quote_fresh": quote_fresh, "book_age_seconds": book_age,
            "book_fresh": book_fresh, "volume": state.volume,
            "open_interest": state.open_interest,
        }
        for window in self.config.probability_windows:
            features[f"prob_change_{window}s"] = state.probability.change(timestamp, window)
        for window in self.config.velocity_windows:
            features[f"prob_velocity_{window}s"] = state.probability.velocity(timestamp, window)
        for window in self.config.acceleration_windows:
            history = state.velocity_histories.get(window)
            features[f"prob_acceleration_{window}s"] = (None if history is None else
                                                          history.velocity(timestamp, window))
        features.update(self._price_features(state, timestamp))
        features.update(self._book_features(state, book_fresh))
        features.update(self._trade_features(state, timestamp))
        features.update(self._time_features(state, timestamp, probability is not None))
        features.update(deepcopy(state.prior_window))
        basket = self.basket_features(timestamp)
        features.update(self._prior_basket_features())
        features.update(self._btc_features(asset, features, basket))
        features["regime_label"] = classify_regime(basket, self.config)
        features["reset_regime_label"] = reset_regime_label(
            self._prior_basket_direction(), basket)
        return {
            "timestamp": timestamp, "asset": asset,
            "market_ticker": state.contract.ticker,
            "contract_open": state.contract.open_time,
            "contract_close": state.contract.close_time,
            "target": state.contract.target,
            "raw_watermark_event_id": watermark,
            "raw_event_ordinal": self.event_count,
            "features": features, "basket": basket,
        }

    def _price_features(self, state, timestamp):
        result = {}
        current = state.price.values[-1][1] if state.price.values else None
        for window in self.config.price_windows:
            prior = state.price.at_or_before(timestamp - window)
            result[f"price_return_{window}s"] = (None if current is None or prior is None
                                                    else safe_ratio(current - prior[1], prior[1]))
        samples = state.price.since(timestamp - self.config.realized_volatility_window)
        prices = [value for _, value in samples]
        log_returns = [log_return(prices[i - 1], prices[i]) for i in range(1, len(prices))]
        log_returns = [value for value in log_returns if value is not None]
        realized = math.sqrt(sum(value * value for value in log_returns)) if log_returns else None
        expected_dollars = None if realized is None or current is None else current * realized
        target = state.contract.target
        absolute = None if current is None or target is None else current - target
        result.update({
            "reference_price": current,
            "realized_volatility_300s": realized,
            "short_horizon_jump_magnitude": (None if not log_returns else abs(log_returns[-1])),
            "rolling_high_300s": max(prices) if prices else None,
            "rolling_low_300s": min(prices) if prices else None,
            "distance_from_rolling_high": (None if not prices else current - max(prices)),
            "distance_from_rolling_low": (None if not prices else current - min(prices)),
            "absolute_target_distance": absolute,
            "percentage_target_distance": (None if absolute is None or target == 0 else absolute / target),
            "volatility_adjusted_target_distance": safe_ratio(absolute, expected_dollars),
        })
        return result

    def _book_features(self, state, fresh):
        if not fresh:
            return {key: None for key in (
                "book_l1_imbalance", "book_top3_imbalance", "book_top5_imbalance",
                "book_bid_depth", "book_ask_depth", "book_depth_ratio", "book_slope")}
        bids = sorted(state.bid_book.items(), reverse=True)
        asks = sorted(state.ask_book.items())
        def imbalance(n):
            bid = sum(qty for _, qty in bids[:n]); ask = sum(qty for _, qty in asks[:n])
            return safe_ratio(bid - ask, bid + ask)
        bid_depth = sum(qty for _, qty in bids[:5])
        ask_depth = sum(qty for _, qty in asks[:5])
        slope = None
        if len(bids) >= 2 and len(asks) >= 2:
            slope = ((bids[0][0] - bids[min(4, len(bids)-1)][0]) +
                     (asks[min(4, len(asks)-1)][0] - asks[0][0]))
        return {"book_l1_imbalance": imbalance(1), "book_top3_imbalance": imbalance(3),
                "book_top5_imbalance": imbalance(5), "book_bid_depth": bid_depth,
                "book_ask_depth": ask_depth, "book_depth_ratio": safe_ratio(bid_depth, ask_depth),
                "book_slope": slope}

    def _trade_features(self, state, timestamp):
        result = {}
        for window in self.config.trade_windows:
            trades = [trade for trade in state.trades if trade[0] >= timestamp - window]
            counts = [trade[1] for trade in trades]
            known = [trade for trade in trades if trade[2] in ("yes", "no")]
            yes = sum(trade[1] for trade in known if trade[2] == "yes")
            no = sum(trade[1] for trade in known if trade[2] == "no")
            half = timestamp - window / 2
            recent = sum(trade[1] for trade in trades if trade[0] >= half) / (window / 2)
            earlier = sum(trade[1] for trade in trades if trade[0] < half) / (window / 2)
            prefix = f"trade_{window}s_"
            result.update({prefix + "count": len(trades), prefix + "contracts": sum(counts),
                           prefix + "yes_aggressive": yes if known else None,
                           prefix + "no_aggressive": no if known else None,
                           prefix + "imbalance": (safe_ratio(yes - no, yes + no) if known else None),
                           prefix + "average_size": mean(counts), prefix + "max_size": max(counts) if counts else None,
                           prefix + "volume_acceleration": recent - earlier})
        return result

    def _time_features(self, state, timestamp, state_valid):
        since = None if state.contract.open_time is None else timestamp - state.contract.open_time
        remaining = None if state.contract.close_time is None else state.contract.close_time - timestamp
        flags = {}
        if since is not None:
            for checkpoint in self.config.checkpoints:
                due = (state_valid and since >= checkpoint and
                       checkpoint not in state.checkpoints_emitted)
                flags[f"checkpoint_{checkpoint}s"] = due
                if due:
                    state.checkpoints_emitted.add(checkpoint)
        return {"seconds_since_contract_open": since,
                "seconds_to_contract_close": remaining, **flags}

    def basket_features(self, timestamp):
        probabilities, velocities, accelerations = {}, {}, {}
        for asset, state in self.assets.items():
            quote_age = None if state.quote_time is None else timestamp - state.quote_time
            p = canonical_probability(state.yes_bid, state.yes_ask, state.no_bid, state.no_ask)[0]
            if quote_age is None or quote_age > self.config.quote_stale_seconds:
                p = None
            probabilities[asset] = p
            velocities[asset] = state.probability.velocity(timestamp, 30)
            accelerations[asset] = state.velocity_histories[30].velocity(timestamp, 30)
        valid_p = [value for value in probabilities.values() if value is not None]
        valid_v = [value for value in velocities.values() if value is not None]
        valid_a = [value for value in accelerations.values() if value is not None]
        above = sum(value > .5 + self.config.neutral_band for value in valid_p)
        below = sum(value < .5 - self.config.neutral_band for value in valid_p)
        neutral = len(valid_p) - above - below
        direction = (f"{above}/5 UP" if above >= 4 else f"{below}/5 DOWN" if below >= 4 else
                     "NEUTRAL" if neutral == len(valid_p) and valid_p else "MIXED")
        signs = [1 if value > self.config.synchronization_velocity_tolerance else
                 -1 if value < -self.config.synchronization_velocity_tolerance else 0
                 for value in valid_v]
        sync = max(signs.count(1), signs.count(-1), signs.count(0)) if signs else 0
        p_std, v_std = stddev(valid_p), stddev(valid_v)
        normalization = mean([math.sqrt(max(value * (1-value), 1e-9)) for value in valid_p])
        normalized_dispersion = safe_ratio(p_std, normalization)
        alt_p = [probabilities[a] for a in ASSET_SERIES if a != "BTC" and probabilities[a] is not None]
        alt_v = [velocities[a] for a in ASSET_SERIES if a != "BTC" and velocities[a] is not None]
        return {
            "probabilities": probabilities, "velocities": velocities,
            "accelerations": accelerations,
            "assets_above_0_50": above, "assets_below_0_50": below,
            "assets_neutral": neutral, "mean_up_probability": mean(valid_p),
            "median_up_probability": median(valid_p), "mean_probability_velocity": mean(valid_v),
            "median_probability_velocity": median(valid_v),
            "mean_probability_acceleration": mean(valid_a), "breadth_direction": direction,
            "synchronized_assets_count": sync, "probability_stddev": p_std,
            "velocity_stddev": v_std, "probability_range": (max(valid_p)-min(valid_p) if valid_p else None),
            "probability_mad": mad(valid_p), "velocity_mad": mad(valid_v),
            "normalized_dispersion_score": normalized_dispersion,
            "dispersion_band": (None if normalized_dispersion is None else
                                "LOW" if normalized_dispersion <= self.config.low_dispersion_threshold else "HIGH"),
            "basket_ex_btc_probability": mean(alt_p), "basket_ex_btc_velocity": mean(alt_v),
            "btc_probability_minus_alt_basket": (None if probabilities["BTC"] is None or not alt_p else
                                                  probabilities["BTC"] - mean(alt_p)),
            "btc_velocity_minus_alt_basket": (None if velocities["BTC"] is None or not alt_v else
                                               velocities["BTC"] - mean(alt_v)),
        }

    def _btc_features(self, asset, features, basket):
        btc_p = basket["probabilities"].get("BTC")
        btc_v = basket["velocities"].get("BTC")
        p = features.get("midpoint_up_probability")
        v = features.get("prob_velocity_30s")
        btc_z = self._current_target_z("BTC")
        z = features.get("volatility_adjusted_target_distance")
        return {
            "alt_probability_minus_btc": None if asset == "BTC" or p is None or btc_p is None else p - btc_p,
            "alt_velocity_minus_btc": None if asset == "BTC" or v is None or btc_v is None else v - btc_v,
            "alt_target_distance_relative_to_btc_factor": (None if asset == "BTC" or z is None or btc_z is None
                                                            else z - btc_z),
        }

    def _current_target_z(self, asset):
        state = self.assets[asset]
        if not state.price.values:
            return None
        return self._price_features(state, self.last_event_time).get("volatility_adjusted_target_distance")

    def _prior_basket_direction(self):
        values = [state.prior_window.get("prior_final_probability") for state in self.assets.values()]
        if any(value is None for value in values):
            return None
        up = sum(value > .5 + self.config.neutral_band for value in values)
        down = sum(value < .5 - self.config.neutral_band for value in values)
        return "UP" if up >= 4 else "DOWN" if down >= 4 else "MIXED"

    def _prior_basket_features(self):
        priors = [state.prior_window for state in self.assets.values()]
        probabilities = [item.get("prior_final_probability") for item in priors]
        returns = [item.get("prior_return") for item in priors]
        valid_p = [value for value in probabilities if value is not None]
        valid_r = [value for value in returns if value is not None]
        up = sum(value > .5 + self.config.neutral_band for value in valid_p)
        down = sum(value < .5 - self.config.neutral_band for value in valid_p)
        direction = "UP" if up == 5 else "DOWN" if down == 5 else "MIXED"
        four = "UP" if up >= 4 else "DOWN" if down >= 4 else "MIXED"
        return {
            "prior_breadth": {"up": up, "down": down, "valid": len(valid_p)},
            "prior_5of5_direction": direction,
            "prior_4of5_direction": four,
            "prior_basket_return": mean(valid_r),
            "prior_basket_extremity": mean([abs(value - .5) for value in valid_p]),
        }


def classify_regime(basket, config):
    if len([p for p in basket["probabilities"].values() if p is not None]) < 5:
        return "NEUTRAL"
    up, down = basket["assets_above_0_50"], basket["assets_below_0_50"]
    if max(up, down) < config.synchronized_min_assets:
        return "NEUTRAL" if basket["assets_neutral"] >= 4 else "FRAGMENTED"
    mean_v = basket["mean_probability_velocity"]
    mean_a = basket["mean_probability_acceleration"]
    if mean_v is None or mean_a is None:
        return "SYNC_UP_DECELERATING" if up >= 4 else "SYNC_DOWN_DECELERATING"
    if up >= 4:
        return "SYNC_UP_ACCELERATING" if mean_a > config.acceleration_epsilon else "SYNC_UP_DECELERATING"
    return "SYNC_DOWN_ACCELERATING" if mean_a < -config.acceleration_epsilon else "SYNC_DOWN_DECELERATING"


def reset_regime_label(prior_direction, basket):
    current = "UP" if basket["assets_above_0_50"] >= 4 else "DOWN" if basket["assets_below_0_50"] >= 4 else "MIXED"
    if current == "MIXED" or prior_direction is None:
        return "FRAGMENTED"
    if prior_direction == current:
        return f"RESET_{current}_CANDIDATE"
    if prior_direction in ("UP", "DOWN"):
        return f"REVERSAL_{current}_CANDIDATE"
    return "FRAGMENTED"
