from typing import Optional

from poke_worlds.emulation.pokemon.parsers import PokemonRedStateParser
from poke_worlds.emulation.tracker import TerminationTruncationMetric, TerminationMetric
from poke_worlds.emulation.pokemon.base_metrics import (
    PokemonExitBattleTruncationMetric,
    PokemonRegionMatchTerminationMetric,
    PokemonRegionMatchTruncationMetric,
)


class PokemonCenterTerminateMetric(
    PokemonRegionMatchTerminationMetric, TerminationMetric
):
    REQUIRED_PARSER = PokemonRedStateParser

    _TERMINATION_NAMED_REGION = "screen_bottom_half"
    _TERMINATION_TARGET_NAME = "viridian_pokemon_center_entrance"
