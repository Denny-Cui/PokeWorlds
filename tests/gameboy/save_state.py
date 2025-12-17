from poke_env import BasicPokemonRedEmulator
import click


@click.command()
@click.option("--sav_file", type=str, default=None, help="Path to save the .sav file")
@click.option("--state_file", type=str, default="tmp.state", help="Path to save the .state file")
def main(sav_file, state_file):
    env = BasicPokemonRedEmulator(parameters=None, headless=True)
    env._sav_to_state(sav_file=sav_file, state_file=state_file)

if __name__ == "__main__":
    main()