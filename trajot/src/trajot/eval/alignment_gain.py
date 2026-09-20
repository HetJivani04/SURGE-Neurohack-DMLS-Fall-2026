from __future__ import annotations


def alignment_gain(aligned_accuracy: float, noalign_accuracy: float) -> float:
    return float(aligned_accuracy - noalign_accuracy)
