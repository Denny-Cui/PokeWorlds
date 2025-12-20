from poke_worlds.emulation.pokemon import PokemonEmulator
from poke_worlds.emulation.pokemon.trackers import PokemonRedStarterTracker
from poke_worlds.interface.environment import DummyEnvironment


class PokemonEnvironment(DummyEnvironment):
    REQUIRED_EMULATOR = PokemonEmulator


class PokemonRedChooseCharmanderFastEnv(PokemonEnvironment):
    REQUIRED_TRACKER = PokemonRedStarterTracker

    def override_emulator_kwargs(emulator_kwargs: dict) -> dict:
        """
        Override default emulator keyword arguments for this environment.
        """
        emulator_kwargs["state_tracker_class"] = PokemonRedStarterTracker
        emulator_kwargs["init_state"] = "starter"
        return emulator_kwargs

    def determine_terminated(self, state):
        starter_chosen = state["pokemon_red_starter"]["current_starter"]
        return starter_chosen is not None
    
    def determine_reward(self, start_state, action, transition_states, action_success):
        current_state = transition_states[-1]
        starter_chosen = current_state["pokemon_red_starter"]["current_starter"]
        n_steps = current_state["core"]["steps"]
        if starter_chosen is None:
            if n_steps >= self._emulator.max_steps-2: # some safety
                return -5.0 # Penalty for not choosing a starter within max steps
            return 0.0 # No reward yet
        if starter_chosen == "charmander":
            step_bonus = 100 / (n_steps+1)
            return 2.0 + step_bonus
        else:
            return -1.0 # Penalty for choosing the wrong starter

