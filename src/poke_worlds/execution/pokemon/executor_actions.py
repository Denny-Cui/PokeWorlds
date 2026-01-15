from poke_worlds.execution.executor_action import ExecutorAction, LocateAction
from poke_worlds.emulation.pokemon import AgentState
from poke_worlds.execution.vlm import ExecutorVLM


class PokemonLocateAction(LocateAction):
    pre_described_options = {
        "item": "a pixelated, greyscale Poke Ball sprite, recognizable by its circular shape, white center, black band around the top, and grey body",
        "pokeball": "a pixelated, greyscale Poke Ball sprite, recognizable by its circular shape, white center, black band around the top, and grey body",
        "npc": "a pixelated human-like character sprite",
        "grass": "a pixelated, greyscale patch of grass that resembles wavy dark lines.",
        "sign": "a pixelated, greyscale white signpost with dots on its face"
    }    


    image_references = {
        "item": "pokeball",
        "pokeball": "pokeball",
        "grass": "grass",
        "sign": "sign"
    }

    def is_valid(self, target = None):
        if self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) != AgentState.FREE_ROAM:
            return False
        return super().is_valid(target=target)

"""
all_options = set(LocateAction.image_references.keys()).union(LocateAction.pre_described_options.keys())
"""


class CheckInteractionAction(ExecutorAction):
    """
    Checks whether a target object is in the interaction sphere of the agent.
    Uses VLM inference to check each grid cell for the target.
    1. Checks the orientation of the agent using VLM inference.
    2. Captures the grid cell in front of the agent and uses VLM inference to describe what is in the cell.

    Is Valid When: 
    - In Free Roam State

    Action Success Interpretation:
    - -1: VLM failure: Could not parse yes or no from response
    - 0: There is something to interact with in front of the agent.
    - 1: There is nothing to interact with in front of the agent.

    Action Returns:
    - `orientation` (`Tuple[int, int]`): The orientation of the agent.
    - `percieve_output` (`str`): The output of the percieve prompt.

    """
    orientation_prompt = """
    You are playing Pokemon and are given a screen capture of the player. Which direction is the player facing?
    Do not give any explanation, just your answer. 
    Answer with one of: UP, DOWN, LEFT, RIGHT and then [STOP]
    Output:
    """

    percieve_prompt = """
    You are playing Pokemon and are given a screen capture of the grid cell in front of the player. 
    Briefly describe what you see in the image, is it an item, or NPC that can be interacted with? Or is it a door or cave that can be entered? If the cell seems empty (or a background texture), say so.
    Give your answer in the following format:
    Description: <a single sentence description of the cell>
    Answer: <YES (if there is something to interact with) or NO (if there is nothing to interact with)>
    and then [STOP]
    Description:
    """
    def is_valid(self, **kwargs):
        return self._state_tracker.get_episode_metric(("pokemon_core", "agent_state")) == AgentState.FREE_ROAM
    
    def _execute(self):
        current_frame = self._emulator.get_current_frame()
        grid_cells = self._emulator.state_parser.capture_grid_cells(current_frame=current_frame)

        orientation_output = ExecutorVLM.infer(texts=[self.orientation_prompt], images=[grid_cells[(0, 0)]], max_new_tokens=5)[0]
        cardinal = None
        if "up" in orientation_output.lower():
            cardinal = (0, 1)
        elif "down" in orientation_output.lower():
            cardinal = (0, -1)
        elif "left" in orientation_output.lower():
            cardinal = (-1, 0)
        elif "right" in orientation_output.lower():
            cardinal = (1, 0)
        if cardinal is None:
            return {"orientation": None, "percieve_output": None}, -1 # This should not happen. It is a VLM failure. 
        cardinal_screen = grid_cells[cardinal]
        percieve_output = ExecutorVLM.infer(texts=[self.percieve_prompt], images=[cardinal_screen], max_new_tokens=50)[0]
        action_success = 0
        if "answer: yes".lower() in percieve_output.lower():
            percieve_output = percieve_output + " you can use interact() now"
        elif "answer: no".lower() in percieve_output.lower():
            action_success = 1
            percieve_output += "You might not be able to use interact() now. But this is not guaranteed."
        else:
            action_success = -1
        ret_dict = {"orientation": cardinal, "percieve_output": percieve_output}
        return ret_dict, action_success
