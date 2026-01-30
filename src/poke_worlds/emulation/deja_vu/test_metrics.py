from typing import Optional
import numpy as np

from poke_worlds.emulation.deja_vu.parsers import DejaVuStateParser
from poke_worlds.emulation.tracker import (
    RegionMatchTerminationMetric,
    TerminationMetric,
)

# Deja Vu Specific Termination Metrics

class DejaVuEnterCastleTerminateMetric(RegionMatchTerminationMetric, TerminationMetric):
    """Ends episode when the agent enters the castle."""
    REQUIRED_PARSER = DejaVuStateParser

    _TERMINATION_NAMED_REGION = "screen_bottom_half"
    _TERMINATION_TARGET_NAME = "enter_castle"


class DejaVuSolveFirstCaseTerminateMetric(RegionMatchTerminationMetric, TerminationMetric):
    """Ends episode when the agent solves the first case."""
    REQUIRED_PARSER = DejaVuStateParser

    _TERMINATION_NAMED_REGION = "dialogue_box_middle"
    _TERMINATION_TARGET_NAME = "solve_first_case"


class DejaVuFindFirstClueTerminateMetric(RegionMatchTerminationMetric, TerminationMetric):
    """Ends episode when the agent finds the first clue."""
    REQUIRED_PARSER = DejaVuStateParser

    _TERMINATION_NAMED_REGION = "dialogue_box_middle"
    _TERMINATION_TARGET_NAME = "find_first_clue"


class DejaVuTalkToCharacterTerminateMetric(RegionMatchTerminationMetric, TerminationMetric):
    """Ends episode when the agent talks to a specific character."""
    REQUIRED_PARSER = DejaVuStateParser

    _TERMINATION_NAMED_REGION = "dialogue_box_middle"
    _TERMINATION_TARGET_NAME = "talk_to_character"


class DejaVuVisitLocationTerminateMetric(RegionMatchTerminationMetric, TerminationMetric):
    """Ends episode when the agent visits a specific location."""
    REQUIRED_PARSER = DejaVuStateParser

    _TERMINATION_NAMED_REGION = "screen_bottom_half"
    _TERMINATION_TARGET_NAME = "visit_location"
