"""Shared risk-bucket policy for Python validation and warehouse exports."""
import numpy as np


def assign_risk_bucket(prob: float) -> str:
    """Apply the fixed descriptive thresholds to a finite probability."""
    if not np.isfinite(prob) or not 0 <= prob <= 1:
        raise ValueError('Risk bucket requires a finite probability in [0, 1]')
    if prob >= 0.70:
        return 'CRITICAL'
    if prob >= 0.40:
        return 'HIGH'
    if prob >= 0.15:
        return 'MEDIUM'
    return 'LOW'
