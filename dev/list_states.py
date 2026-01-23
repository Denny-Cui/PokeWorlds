import click
from poke_worlds import AVAILABLE_GAMES
from poke_worlds.utils import load_parameters, log_info, log_warn, log_dict
import os
import pandas as pd


@click.command()
@click.option(
    "--game",
    type=click.Choice(AVAILABLE_GAMES),
    required=True,
    help="The game to list states for.",
)
def list_states(game):
    """Lists all available states for a given game."""
    params = load_parameters()
    state_path = params[f"{game}_rom_data_path"] + "/states/"
    if not os.path.exists(state_path):
        log_warn(f"{state_path} does not exist.", params)
        return
    states = os.listdir(state_path)
    if len(states) == 0:
        log_warn(f"No states found in {state_path}.", params)
        return
    state_dict = {"base": []}
    for state_file in states:
        if state_file.endswith(".state"):
            state_name = state_file[:-6]
            if "_" in state_name:
                base_name = state_name.split("_")[0]
            else:
                base_name = "base"
            if base_name not in state_dict:
                state_dict[base_name] = []
            state_dict[base_name].append(state_name)
    for key in state_dict:
        state_dict[key].sort()
    format_str = f"Available states for game {game}:"
    for key in state_dict:
        format_str += f"\n  {key}:\n"
        for state_name in state_dict[key]:
            format_str += f"    - {state_name}\n"
    log_info(format_str, params)
    save_path = "tmp_state_list.csv"
    columns = ["name"]
    data = []
    for key in state_dict:
        for state_name in state_dict[key]:
            data.append([state_name])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path, index=False)
    log_info(f"Saved state list to {save_path}", params)


if __name__ == "__main__":
    list_states()
