from poke_worlds.emulation.pokemon.base_metrics import (
    CorePokemonMetrics,
    PokemonOCRMetric,
    PokemonRedLocation,
    PokemonRedStarter,
)
from poke_worlds.emulation.pokemon.base_metrics import (
    PokemonTestMetric,
)
from poke_worlds.utils import log_info
from poke_worlds.emulation.tracker import (
    StateTracker,
)
from poke_worlds.emulation.pokemon.parsers import (
    AgentState,
)
from typing import Optional


class CorePokemonTracker(StateTracker):
    """
    StateTracker for core Pokémon metrics.
    """

    def start(self):
        super().start()
        self.metric_classes.extend([CorePokemonMetrics, PokemonTestMetric])

    def step(self, *args, **kwargs):
        """
        Calls on super().step(), but then modifies the current frame to overlay the grid if the agent is in FREE ROAM.
        """
        super().step(*args, **kwargs)
        state = self.episode_metrics["pokemon_core"]["agent_state"]
        # if agent_state is in FREE ROAM, draw the grid, otherwise do not
        if state == AgentState.FREE_ROAM:
            screen = self.episode_metrics["core"]["current_frame"]
            screen = self.state_parser.draw_grid_overlay(current_frame=screen)
            self.episode_metrics["core"]["current_frame"] = screen
            previous_screens = self.episode_metrics["core"]["passed_frames"]
            if previous_screens is not None:
                self.episode_metrics["core"]["passed_frames"][-1, :] = screen


class PokemonOCRTracker(CorePokemonTracker):
    def start(self):
        super().start()
        self.metric_classes.extend([PokemonOCRMetric])


class PokemonRedStarterTracker(CorePokemonTracker):
    """
    Example StateTracker that tracks the starter Pokémon chosen in Pokémon Red.
    """

    def start(self):
        super().start()
        self.metric_classes.extend([PokemonRedStarter, PokemonRedLocation])
