"""Safe model persistence with skops instead of pickle.

Unpickling a file can run arbitrary code, so a tampered model file could
compromise the service. skops stores the model as data and, when loading,
refuses any object type that is not explicitly trusted. The list below is
the complete set of non-standard types in the v2 pipeline; anything else in
a model file makes loading fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import skops.io

TRUSTED_TYPES = [
    "numpy.dtype",
    "readmission_risk.features.build._as_object",
    "readmission_risk.features.build.add_features",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def save(model: Any, path: Path) -> None:
    """Write ``model`` to ``path`` in skops format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    skops.io.dump(model, path)


def load(path: Path) -> Any:
    """Load a model, allowing only :data:`TRUSTED_TYPES` beyond skops defaults.

    Raises:
        skops.io.exceptions.UntrustedTypesFoundException: If the file contains
            any other non-default type.
    """
    return skops.io.load(path, trusted=TRUSTED_TYPES)
