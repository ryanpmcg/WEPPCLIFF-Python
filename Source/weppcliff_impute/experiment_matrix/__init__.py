"""Ampute–impute experiment matrix (paper protocol A–F + dual whole/hole scoring).

See ``Documentation/Planning/AmputeImputeExperimentMatrix.md`` for the full design.
"""

from Source.weppcliff_impute.experiment_matrix.catalog import (
    EXPERIMENT_CATALOG,
    ExperimentSpec,
    list_experiments,
)
from Source.weppcliff_impute.experiment_matrix.runner import (
    MatrixRunConfig,
    run_experiment,
    run_matrix,
)
from Source.weppcliff_impute.experiment_matrix.scoring import score_dual_panel

__all__ = [
    "EXPERIMENT_CATALOG",
    "ExperimentSpec",
    "MatrixRunConfig",
    "list_experiments",
    "run_experiment",
    "run_matrix",
    "score_dual_panel",
]
