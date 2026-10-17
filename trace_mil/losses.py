"""Cox partial likelihood. The model outputs one risk per patient: a HIGHER value means a HIGHER risk (shorter survival)."""
from __future__ import annotations

import torch


def cox_ph_loss(risk_score: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow approximation for ties).

    risk_score: (N,) higher = higher risk (shorter survival).
    time:       (N,) observed follow-up time (event time if event=1, else
                censoring time).
    event:      (N,) 1 if the event was observed, 0 if censored.

    For each subject i with event_i=1, the risk set R(t_i) = {j: time_j >=
    time_i} is exactly the prefix of the time-descending-sorted array up to
    and including i, so a single cumulative logsumexp over that ordering
    gives log(sum_{j in R(t_i)} exp(risk_j)) for every i at once.
    """
    order = torch.argsort(time, descending=True)
    risk_sorted = risk_score[order]
    event_sorted = event[order]

    log_cumsum_risk = torch.logcumsumexp(risk_sorted, dim=0)
    log_lik = (risk_sorted - log_cumsum_risk) * event_sorted

    num_events = event_sorted.sum()
    if num_events.item() == 0:
        # No events in this batch: partial likelihood is undefined: return a
        # zero loss that still participates in the autograd graph.
        return risk_score.sum() * 0.0
    return -log_lik.sum() / num_events
