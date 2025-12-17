from poke_env import get_pokemon_emulator, AVAILABLE_POKEMON_VARIANTS
import click


@click.command()
@click.option("--variant", type=click.Choice(AVAILABLE_POKEMON_VARIANTS), default="pokemon_red", help="Variant of the Pokemon game to emulate.")
@click.option("--sav_file", type=str, default=None, help="Path to save the .sav file")
@click.option("--state_file", type=str, default="tmp.state", help="Path to save the .state file")
def main(variant, sav_file, state_file):
    env = get_pokemon_emulator(variant=variant, headless=False)
    env._sav_to_state(sav_file=sav_file, state_file=state_file)

if __name__ == "__main__":
    main()