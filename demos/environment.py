from poke_worlds import AVAILABLE_POKEMON_VARIANTS, get_pokemon_environment, LowLevelController, RandomPlayController, LowLevelPlayController
from tqdm import tqdm
import click


@click.command()
@click.option("--variant", type=click.Choice(AVAILABLE_POKEMON_VARIANTS), default="pokemon_red", help="Variant of the Pokemon game to emulate.")
@click.option("--init_state", type=str, default=None, help="Name of the initial state file")
@click.option("--play_mode", type=click.Choice(["random", "restricted_random", "grouped_random"]), default="random", help="Play mode: 'random' for random actions.")
@click.option("--render", type=bool, default=False, help="Whether to render the environment with PyGame.")
@click.option("--save_video", type=bool, default=None, help="Whether to save a video of the gameplay. If not specified, uses default from config.")
def main(variant, init_state, play_mode, render, save_video):
    if play_mode == "random":
        controller = LowLevelController()
    elif play_mode == "restricted_random":
        controller = LowLevelPlayController()
    elif play_mode == "grouped_random":
        controller = RandomPlayController()
    else:
        raise ValueError(f"Unknown play mode: {play_mode}")
    environment = get_pokemon_environment(game_variant=variant, controller=controller, headless=True, save_video=save_video, init_state=init_state)
    steps = 0
    max_steps = 500
    pbar = tqdm(total=max_steps)
    while steps < max_steps:
        action = environment.action_space.sample()
        observation, reward, terminated, truncated, info = environment.step(action)
        if render:
            environment.render()
        if terminated or truncated:
            break
        steps += 1
        pbar.update(1)
    pbar.close()    
    environment.close()

if __name__ == "__main__":
    main()