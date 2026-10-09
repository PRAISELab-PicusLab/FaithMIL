"""From a Cox risk score to the challenge's time-to-recurrence and risk group.

The network is trained with the Cox partial likelihood (src/losses.py), which
only identifies the risk ORDER. Absolute predictions come from the Breslow
estimate of the baseline cumulative hazard on the training fold:

    S(t | r) = exp(-H0(t) * exp(r - r_offset))

Time to recurrence = restricted mean survival time RMST(tau) = int_0^tau S(t|r) dt.
It is finite even when S never drops below 0.5 (5-year BCR-free survival is
86% here, so a median time is undefined for most patients) and it is strictly
decreasing in r, so the C-index of the predicted times equals the C-index of
the risk scores.

Risk group ('low' / 'medium' / 'high') thresholds are quantiles of the
predicted times on the training fold, as the challenge asks for groups derived
from the time-to-recurrence prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BreslowModel:
    event_times: np.ndarray  # unique event times, ascending
    cum_hazard: np.ndarray   # H0 at event_times, for risk centred by risk_offset
    risk_offset: float
    horizon: float           # RMST horizon tau (years)

    def survival_at(self, risk, times) -> np.ndarray:
        """S(t | r) for every risk (rows) and time (columns)."""
        risk = np.atleast_1d(np.asarray(risk, dtype=np.float64))
        idx = np.searchsorted(self.event_times, np.asarray(times, dtype=np.float64), side="right") - 1
        H = np.where(idx >= 0, self.cum_hazard[np.clip(idx, 0, None)], 0.0)
        return np.exp(-H[None, :] * np.exp(risk - self.risk_offset)[:, None])

    def rmst(self, risk) -> np.ndarray:
        risk = np.atleast_1d(np.asarray(risk, dtype=np.float64))
        t = self.event_times[self.event_times < self.horizon]
        knots = np.concatenate([[0.0], t, [self.horizon]])
        H_left = np.concatenate([[0.0], self.cum_hazard[: len(t)]])  # H0 on [knot_j, knot_j+1)
        S_left = np.exp(-H_left[None, :] * np.exp(risk - self.risk_offset)[:, None])
        return (S_left * np.diff(knots)[None, :]).sum(axis=1)

    def median_time(self, risk) -> np.ndarray:
        """First event time where S <= 0.5 (NaN if never reached)."""
        S = self.survival_at(risk, self.event_times)
        below = S <= 0.5
        return np.where(below.any(axis=1), self.event_times[np.argmax(below, axis=1)], np.nan)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_times"] = self.event_times.tolist()
        d["cum_hazard"] = self.cum_hazard.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BreslowModel":
        return cls(np.asarray(d["event_times"], dtype=np.float64), np.asarray(d["cum_hazard"], dtype=np.float64),
                   float(d["risk_offset"]), float(d["horizon"]))


def fit_breslow(risk, time, event, horizon: float | None = None) -> BreslowModel:
    risk = np.asarray(risk, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event).astype(bool)
    offset = float(risk.mean())
    w = np.exp(risk - offset)

    event_times = np.unique(time[event])
    dH = np.empty(len(event_times))
    for j, t in enumerate(event_times):
        d = np.sum(event & (time == t))
        dH[j] = d / w[time >= t].sum()
    tau = float(time.max()) if horizon is None else float(horizon)
    return BreslowModel(event_times, np.cumsum(dH), offset, tau)


RISK_GROUPS = ("high", "medium", "low")


def risk_group_thresholds(train_times, quantiles=(1 / 3, 2 / 3)) -> list[float]:
    return [float(q) for q in np.quantile(np.asarray(train_times), quantiles)]


def assign_risk_group(times, thresholds) -> np.ndarray:
    """Shortest predicted times -> 'high' risk."""
    lo, hi = thresholds
    times = np.asarray(times)
    return np.where(times <= lo, "high", np.where(times <= hi, "medium", "low"))
