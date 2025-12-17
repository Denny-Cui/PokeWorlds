from poke_env import get_pokemon_emulator, AVAILABLE_POKEMON_VARIANTS
import click


@click.command()
@click.option("--variant", type=click.Choice(AVAILABLE_POKEMON_VARIANTS), default="pokemon_red", help="Variant of the Pokemon game to emulate.")
@click.option("--init_state", type=str, default=None, help="Name of the initial state file")
@click.option("--play_mode", type=click.Choice(["human", "random"]), default="human", help="Play mode: 'human' for manual play, 'random' for random actions.")
def main(variant, init_state, play_mode):
    env = get_pokemon_emulator(variant=variant, init_state_name=init_state, headless=False)
    if play_mode == "human":
        env.human_play()
    else:
        raise NotImplementedError("Only human play mode is implemented in this demo.")

if __name__ == "__main__":
    main()