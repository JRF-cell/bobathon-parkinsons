"""Evaluation strategy for experiment 01_dummy.

Per the lab guide, 01_dummy uses a simple random 80/20 visit-level
holdout (splitter=0.2). The same patient can appear on both sides of
the split — this is intentional for this baseline to illustrate the
visit-level leakage that grouped CV will fix in a later experiment.
"""

from __future__ import annotations

# 80/20 random visit-level holdout.
# skore interprets a float as train_size = 1 - splitter, test_size = splitter.
splitter = 0.2
