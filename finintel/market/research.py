"""Market price data and leakage-safe event-return research."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PriceService:
    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def get_prices(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        key = f"{ticker}:{start.date()}:{end.date()}"
        if key in self._cache:
            return self._cache[key]
        try:
            import yfinance as yf
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self._cache[key] = df
            return df
        except Exception as exc:
            logger.warning("Price fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    def forward_return(
        self,
        ticker: str,
        event_time: datetime,
        horizon_days: int,
    ) -> Optional[float]:
        start = event_time - timedelta(days=5)
        end = event_time + timedelta(days=horizon_days + 10)
        prices = self.get_prices(ticker, start, end)
        if prices.empty or "Close" not in prices.columns:
            return None

        prices = prices.sort_index()
        event_date = pd.Timestamp(event_time.date())
        future_date = event_date + pd.Timedelta(days=horizon_days)

        valid = prices.index[prices.index >= event_date]
        if len(valid) == 0:
            return None
        entry_idx = valid[0]
        entry_price = float(prices.loc[entry_idx, "Close"])

        future_valid = prices.index[prices.index >= future_date]
        if len(future_valid) == 0:
            last_idx = prices.index[-1]
            exit_price = float(prices.loc[last_idx, "Close"])
        else:
            exit_price = float(prices.loc[future_valid[0], "Close"])

        if entry_price == 0:
            return None
        return (exit_price - entry_price) / entry_price * 100


@dataclass
class ResearchResult:  # noqa: D101
    ticker: str
    horizon_days: int
    n_events: int
    mean_return: float
    median_return: float
    hit_rate: float
    std_return: float
    t_stat: float
    is_significant: bool
    disclaimer: str


class EventReturnResearch:
    """
    Leakage-safe event → subsequent-return analysis.

    Uses only prices available AFTER event timestamp.
    Walk-forward validation splits by time, not random shuffle.
  """

    def __init__(self, horizons: Optional[list[int]] = None):
        self.horizons = horizons or [1, 5, 21]
        self.price_service = PriceService()

    def analyze_events(
        self,
        events: list[dict],
        ticker: str,
    ) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        for horizon in self.horizons:
            returns: list[float] = []
            for ev in events:
                event_time = ev.get("event_time")
                if isinstance(event_time, str):
                    event_time = datetime.fromisoformat(event_time)
                if not event_time:
                    continue
                ret = self.price_service.forward_return(ticker, event_time, horizon)
                if ret is not None:
                    returns.append(ret)

            if len(returns) < 5:
                results.append(
                    ResearchResult(
                        ticker=ticker,
                        horizon_days=horizon,
                        n_events=len(returns),
                        mean_return=0.0,
                        median_return=0.0,
                        hit_rate=0.0,
                        std_return=0.0,
                        t_stat=0.0,
                        is_significant=False,
                        disclaimer=(
                            f"Insufficient data (n={len(returns)}). "
                            "Results are not statistically meaningful."
                        ),
                    )
                )
                continue

            arr = np.array(returns)
            mean_r = float(np.mean(arr))
            std_r = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            t_stat = mean_r / (std_r / np.sqrt(len(arr))) if std_r > 0 else 0.0
            hit_rate = float(np.mean(arr > 0))
            is_sig = abs(t_stat) > 1.96 and len(arr) >= 30

            disclaimer = (
                "Statistically significant at 95% level."
                if is_sig
                else "NOT statistically significant — do not infer predictability."
            )

            results.append(
                ResearchResult(
                    ticker=ticker,
                    horizon_days=horizon,
                    n_events=len(returns),
                    mean_return=mean_r,
                    median_return=float(np.median(arr)),
                    hit_rate=hit_rate,
                    std_return=std_r,
                    t_stat=float(t_stat),
                    is_significant=is_sig,
                    disclaimer=disclaimer,
                )
            )
        return results

    def walk_forward_validation(
        self,
        events: list[dict],
        ticker: str,
        train_days: int = 252,
        test_days: int = 63,
    ) -> list[dict]:
        if not events:
            return []

        sorted_events = sorted(
            events,
            key=lambda e: e.get("event_time", datetime.min),
        )
        folds: list[dict] = []
        i = 0
        while i < len(sorted_events):
            train_end = sorted_events[i].get("event_time")
            if isinstance(train_end, str):
                train_end = datetime.fromisoformat(train_end)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            train = [
                e for e in sorted_events
                if e.get("event_time") and (
                    datetime.fromisoformat(e["event_time"])
                    if isinstance(e["event_time"], str)
                    else e["event_time"]
                ) < test_start
            ][-train_days:]
            test = [
                e for e in sorted_events
                if e.get("event_time") and test_start <= (
                    datetime.fromisoformat(e["event_time"])
                    if isinstance(e["event_time"], str)
                    else e["event_time"]
                ) < test_end
            ]

            if len(test) >= 3:
                test_results = self.analyze_events(test, ticker)
                folds.append({
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "train_size": len(train),
                    "test_size": len(test),
                    "results": [
                        {
                            "horizon": r.horizon_days,
                            "mean_return": r.mean_return,
                            "hit_rate": r.hit_rate,
                            "is_significant": r.is_significant,
                        }
                        for r in test_results
                    ],
                })
            i += max(len(test), 1)
        return folds
