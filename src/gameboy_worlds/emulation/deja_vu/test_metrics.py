from gameboy_worlds.emulation.deja_vu.parsers import DejaVu1StateParser
from gameboy_worlds.emulation.tracker import (
    RegionMatchTerminationMetric,
    TerminationMetric,
)


class DejaVuCoatTerminationMetric(RegionMatchTerminationMetric, TerminationMetric):
    REQUIRED_PARSER = DejaVu1StateParser

    _TERMINATION_NAMED_REGION = "dialogue_box_area"
    _TERMINATION_TARGET_NAME = "took_coat"
