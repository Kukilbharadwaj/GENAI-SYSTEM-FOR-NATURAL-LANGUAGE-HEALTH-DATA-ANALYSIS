"""
Evaluation framework for testing pipeline quality
"""

from .evaluation_framework import (
    PipelineEvaluator,
    run_quick_evaluation,
    run_full_evaluation,
    EVALUATION_TEST_CASES
)

__all__ = [
    'PipelineEvaluator',
    'run_quick_evaluation',
    'run_full_evaluation',
    'EVALUATION_TEST_CASES'
]
