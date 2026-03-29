from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

N_ASSETS = 25
TICKS_PER_DAY = 30
ASSET_COLUMNS = tuple(f"A{i:02d}" for i in range(N_ASSETS))


@dataclass(frozen=True)
class PublicMeta:
    sector_id: np.ndarray
    spread_bps: np.ndarray
    borrow_bps_annual: np.ndarray


def load_prices(path: str = "prices.csv") -> np.ndarray:
    df = pd.read_csv(path, index_col="tick")
    return df[list(ASSET_COLUMNS)].to_numpy(dtype=float)


def load_meta(path: str = "meta.csv") -> PublicMeta:
    df = pd.read_csv(path)
    return PublicMeta(
        sector_id=df["sector_id"].to_numpy(dtype=int),
        spread_bps=df["spread_bps"].to_numpy(dtype=float),
        borrow_bps_annual=df["borrow_bps_annual"].to_numpy(dtype=float),
    )


class StrategyBase:
    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        pass

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        raise NotImplementedError


class MyStrategy(StrategyBase):
    def __init__(self) -> None:
        self.prev = np.zeros(N_ASSETS, dtype=float)
        self.lookback = 5
        self.vol_lookback = 20
        self.smooth = 0.20
        self.band = 0.005
        self.short_penalty = 2.0

    def fit(self, train_prices: np.ndarray, meta: PublicMeta, **kwargs) -> None:
        self.prev = np.zeros(N_ASSETS, dtype=float)

    def get_weights(self, price_history: np.ndarray, meta: PublicMeta, day: int) -> np.ndarray:
        closes = price_history[TICKS_PER_DAY - 1 :: TICKS_PER_DAY]
        n_days = closes.shape[0]
        if n_days < max(self.lookback + 1, self.vol_lookback + 2):
            return self.prev.copy()

        daily_returns = closes[1:] / closes[:-1] - 1.0
        momentum = closes[-1] / closes[-1 - self.lookback] - 1.0
        vol = np.std(daily_returns[-self.vol_lookback :], axis=0, ddof=1) + 1e-6

        score = momentum / vol

        # Sector-demean: bet on relative strength within a sector, not broad sector drift.
        for sector in np.unique(meta.sector_id):
            idx = meta.sector_id == sector
            score[idx] -= np.mean(score[idx])

        score_std = np.std(score)
        if score_std > 1e-12:
            score = (score - np.mean(score)) / score_std
        else:
            score = np.zeros_like(score)

        cost_penalty = 1.0 + 8.0 * (meta.spread_bps / 1e4)
        raw = score / vol / cost_penalty
        raw = np.where(raw < 0.0, raw / (1.0 + self.short_penalty * meta.borrow_bps_annual / 1e4), raw)

        gross = float(np.sum(np.abs(raw)))
        target = np.zeros_like(raw) if gross < 1e-12 else raw / gross

        updated = (1.0 - self.smooth) * self.prev + self.smooth * target
        updated = np.where(np.abs(updated - self.prev) < self.band, self.prev, updated)

        gross = float(np.sum(np.abs(updated)))
        if gross > 1.0:
            updated /= gross

        self.prev = updated.copy()
        return updated


def create_strategy() -> StrategyBase:
    return MyStrategy()
