"""Flattened node metrics plus uncertainty-error diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .metrics import evaluate_probabilities


def evaluate_open(y,p,uncertainty,time_ids,bins=15):
    result=evaluate_probabilities(y,p,severe=2,bins=bins)
    error=(p.argmax(1)!=y).astype(float)
    corr,pval=spearmanr(uncertainty,error) if np.ptp(uncertainty)>1e-10 else (np.nan,np.nan)
    result.update({"uncertainty_error_spearman":float(corr),"uncertainty_error_p":float(pval),
                   "n_predictions":len(y),"n_time_blocks":len(np.unique(time_ids))})
    return result
