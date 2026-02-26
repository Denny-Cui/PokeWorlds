"""
DejaVu I & II: The Casebooks of Ace Harding game state parser implementations.

Deja Vu is a detective mystery game focused on investigation and puzzle-solving.
The game mechanics are fundamentally different from Pokemon:
- No battles or combat mechanics
- Heavy emphasis on dialogue, evidence collection, and deduction
- Menu-driven investigation system with case notes, evidence, and location management
- Puzzle-solving segments where player must make logical deductions

This parser provides visual-based state detection for:
1. FREE_ROAM: Walking around investigation areas and locations
2. IN_DIALOGUE: Interacting with NPCs, getting clues, and story progression
3. IN_MENU: Accessing case notes, evidence view, location map, or other menus

CORE DESIGN PRINCIPLE: Never branch the parser subclasses for a given variant. The inheritance tree for a parser after the game variant parser should always be a tree with only one child per layer.
This is to ensure that we don't double effort, any capability added to a parser will always be valid for that game variant.
If this principle is followed, any state tracker can always use the STRONGEST (lowest level) parser for a given variant without concern for missing functionality.
"""

from poke_worlds.emulation.parser import NamedScreenRegion
from poke_worlds.utils import (
    log_warn,
    log_info,
    log_error,
    load_parameters,
    verify_parameters,
)
from poke_worlds.emulation.parser import StateParser

from typing import Set, List, Type, Dict, Optional, Tuple
import os
from abc import ABC, abstractmethod
from enum import Enum

from pyboy import PyBoy

import json
import numpy as np
from bidict import bidict


class AgentState(Enum):
    """
    0. FREE_ROAM: The agent is freely roaming the game world.
    1. IN_DIALOGUE: The agent is currently in a dialogue state. (including reading signs, talking to NPCs, etc.)
    2. IN_MENU: The agent is currently in a menu state. (including PC, Name Entry, Pokedex, etc.)
    """

    FREE_ROAM = 0
    IN_DIALOGUE = 1
    IN_MENU = 2


def _get_proper_regions(
    override_regions: List[Tuple[str, int, int, int, int]],
    base_regions: List[Tuple[str, int, int, int, int]],
) -> List[Tuple[str, int, int, int, int]]:
    """
    Merges base regions with override regions, giving precedence to override regions.

    :param override_regions: List of override region tuples.
    :type override_regions: List[Tuple[str, int, int, int, int]]
    :param base_regions: List of base region tuples.
    :type base_regions: List[Tuple[str, int, int, int, int]]
    :return: Merged list of region tuples.
    :rtype: List[Tuple[str, int, int, int, int]]
    """
    if len(override_regions) == 0:
        return base_regions
    proper_regions = override_regions.copy()
    override_names = [region[0] for region in override_regions]
    for region in base_regions:
        if region[0] in override_names:
            continue
        proper_regions.append(region)
    return proper_regions


class DejaVuStateParser(StateParser, ABC):
    """
    Base class for DejaVu game state parsers. Uses visual screen regions to parse game state.
    Defines common named screen regions and methods for determining game states such as being in battle, menu, or dialogue.

    Can be used to determine the exact AgentState
    """

    COMMON_REGIONS = [
        ("dialogue_top_left_hook", 0, 73, 10, 6),  # Top left hook that appears after certain events. Can be used to determine if certain game mechanics are available.
        ("menu_bottom_line", 0, 143, 160, 1),  # Bottom line that appears when any menu is open, can be used to prevent agent interaction with the UI frame of the emulator.
    ]
    """ List of common named screen regions for Deja Vu game.
    
    Deja Vu uses a primarily text/menu-driven interface. These regions help identify:
    - dialogue_top_left_hook: A hook that appears in the top left after certain events, can be used to determine if certain game mechanics are available.
    - menu_bottom_line: A line that appears at the bottom of the screen when any menu is open, can be used to prevent agent interaction with the UI frame of the emulator.
    """

    # COMMON_MULTI_TARGET_REGIONS = []
    # """ List of common multi-target named screen regions for Deja Vu games."""

    # COMMON_MULTI_TARGETS = {}
    # """ Common multi-targets for Deja Vu game regions."""

    def __init__(
        self,
        variant: str,
        pyboy: PyBoy,
        parameters: dict,
        additional_named_screen_region_details: List[Tuple[str, int, int, int, int]] = [],
        additional_multi_target_named_screen_region_details: List[Tuple[str, int, int, int, int]] = [],
        override_multi_targets: Dict[str, List[str]] = {},
    ):
        """
        Initializes the DejaVuStateParser.
        Args:
            variant (str): The variant of the Deja Vu game.
            pyboy (PyBoy): The PyBoy emulator instance.
            parameters (dict): Configuration parameters for the emulator.
            additional_named_screen_region_details (List[Tuple[str, int, int, int, int]]): Parameters associated with additional named screen regions to include.
            additional_multi_target_named_screen_region_details (List[Tuple[str, int, int, int, int]]): Parameters associated with additional multi-target named screen regions to include.
            override_multi_targets (Dict[str, List[str]]): Dictionary mapping region names to lists of target names for multi-target regions.
                By default, will add "menu_box_strip" with target "cursor_on_options". This is important because we don't want agents messing with the frame of the emulator (it will wreck our state parsing).
        """
        verify_parameters(parameters)
        # regions = _get_proper_regions(
        #     override_regions=additional_named_screen_region_details,
        #     base_regions=self.COMMON_REGIONS,
        # )
        regions = self.COMMON_REGIONS
        if f"{variant}_rom_data_path" not in parameters:
            log_error(
                f"ROM data path not found for variant: {variant}. Add {variant}_rom_data_path to the config files. See configs/deja_vu_vars.yaml for an example",
                parameters,
            )
        self.variant = variant
        self.rom_data_path = parameters[f"{variant}_rom_data_path"]
        """ Path to the ROM data directory for the specific Deja Vu variant."""
        captures_dir = self.rom_data_path + "/captures/"
        named_screen_regions = []
        for region_name, x, y, w, h in regions:
            region = NamedScreenRegion(
                region_name,
                x,
                y,
                w,
                h,
                parameters=parameters,
                target_path=os.path.join(captures_dir, region_name),
            )
            named_screen_regions.append(region)
        # multi_target_regions = _get_proper_regions(
        #     override_regions=additional_multi_target_named_screen_region_details,
        #     base_regions=self.COMMON_MULTI_TARGET_REGIONS,
        # )
        # multi_target_region_names = [region[0] for region in multi_target_regions]
        # multi_targets = self.COMMON_MULTI_TARGETS.copy()
        # for key in override_multi_targets:
        #     if key in multi_targets:
        #         multi_targets[key].extend(override_multi_targets[key])
        #     else:
        #         multi_targets[key] = override_multi_targets[key]
        # multi_target_provided_region_names = list(multi_targets.keys())
        # if not set(multi_target_provided_region_names).issubset(
        #     set(multi_target_region_names)
        # ):
        #     log_error(
        #         f"Multi-target regions provided in multi_targets do not match the defined multi-target regions. Provided: {multi_target_provided_region_names}, Defined: {multi_target_region_names}",
        #         parameters,
        #     )
        # for region_name, x, y, w, h in multi_target_regions:
        #     region_target_paths = {}
        #     subdir = captures_dir + f"/{region_name}/"
        #     for target_name in multi_targets.get(region_name, []):
        #         region_target_paths[target_name] = os.path.join(subdir, target_name)
        #     region = NamedScreenRegion(
        #         region_name,
        #         x,
        #         y,
        #         w,
        #         h,
        #         parameters=parameters,
        #         multi_target_paths=region_target_paths,
        #     )
        #     named_screen_regions.append(region)
        super().__init__(pyboy, parameters, named_screen_regions)

    def is_in_menu(
        self, current_screen: np.ndarray
    ) -> bool:
        """
        Determines if any form of menu is currently open (Case Notes, Evidence, Location, etc).

        Args:
            current_screen (np.ndarray): The current screen frame from the emulator.
            trust_previous (bool): If True, trusts that checks for other states have been done.

        Returns:
            bool: True if a menu is open, False otherwise.
        """
        return self.named_region_matches_target(current_screen, 'menu_bottom_line')

    def is_in_dialogue(
        self, current_screen: np.ndarray
    ) -> bool:
        """
        Determines if the player is currently in a dialogue state.
        Includes talking to NPCs, receiving clues, story narration, etc.

        Args:
            current_screen (np.ndarray): The current screen frame from the emulator.
            trust_previous (bool): If True, trusts that checks for menu state have been done.

        Returns:
            bool: True if in dialogue, False otherwise.
        """
        if self.is_in_menu(current_screen):
            return False
        return self.named_region_matches_target(current_screen, "dialogue_top_left_hook")

    def get_agent_state(self, current_screen: np.ndarray) -> AgentState:
        """
        Determines the current agent state based on the screen.

        Uses trust_previous to optimize checks.

        Args:
            current_screen (np.ndarray): The current screen frame from the emulator.

        Returns:
            AgentState: The current agent state (FREE_ROAM, IN_DIALOGUE, or IN_MENU).
        """
        if self.is_in_menu(current_screen):
            return AgentState.IN_MENU
        elif self.is_in_dialogue(current_screen):
            return AgentState.IN_DIALOGUE
        else:
            return AgentState.FREE_ROAM


class BaseDejaVu1StateParser(DejaVuStateParser, ABC):
    """
    Game state parser for all DejaVu-based games.
    Implements base functionality for identifying investigation UI elements.
    """

    REGIONS = []
    """ Additional named screen regions specific to Deja Vu games.
    
    - case_notes_header_area: Top area when Case Notes menu is open.
    - evidence_list_top: Top area when Evidence/Inventory menu is open.
    - location_map_header: Top area when Location/Map menu is open.
    - suspect_name_region: Area showing suspect/character names during dialogue.
    - clue_received_indicator: Area showing newly received clue notifications.
    """

    MULTI_TARGET_REGIONS = []
    """ Additional multi-target named screen regions specific to Deja Vu games.
    
    - menu_navigation_strip: Top menu bar showing available menu options and current selection.
    """

    def __init__(
        self,
        pyboy: PyBoy,
        variant: str,
        parameters: dict,
        override_regions: List[Tuple[str, int, int, int, int]] = [],
        override_multi_target_regions: List[Tuple[str, int, int, int, int]] = [],
        override_multi_targets: Dict[str, List[str]] = {},
    ):
        self.REGIONS = _get_proper_regions(
            override_regions=override_regions, base_regions=self.REGIONS
        )
        self.MULTI_TARGET_REGIONS = _get_proper_regions(
            override_regions=override_multi_target_regions,
            base_regions=self.MULTI_TARGET_REGIONS,
        )
        super().__init__(
            variant=variant,
            pyboy=pyboy,
            parameters=parameters,
            additional_named_screen_region_details=self.REGIONS,
            additional_multi_target_named_screen_region_details=self.MULTI_TARGET_REGIONS,
            override_multi_targets=override_multi_targets,
        )

    def __repr__(self):
        return f"<DejaVuParser(variant={self.variant})>"

class DejaVu1StateParser(BaseDejaVu1StateParser):
    """
    Game state parser for Deja Vu I: The Casebooks of Ace Harding.
    
    Implements multi-target recognition for specific game events:
    - Case progression (clues acquired, verdicts reached)
    - Investigation phases (suspect interrogations, alibi checks)
    - Case solution states (correct/incorrect conclusions)
    """

    def __init__(self, pyboy, parameters):
        override_multi_targets = {}
        super().__init__(
            pyboy,
            variant="deja_vu_1",
            parameters=parameters,
            override_multi_targets=override_multi_targets,
        )

"""
The below code shows how to add domain information into the game state parser and read from memory addresses to get descriptive state information. 

This is not actually used in any of the current environments, but is left here to show that if you want to bake in more domain knowledge and create explicit reward schedules etc., you can read the information required to do so in this class. 
"""


# class MemoryBasedDejaVuStateParser(DejaVuStateParser):

#     _PAD = 20
#     _GLOBAL_MAP_SHAPE = (444 + _PAD * 2, 436 + _PAD * 2)
#     _MAP_ROW_OFFSET = _PAD
#     _MAP_COL_OFFSET = _PAD

#     def __init__(self, pyboy, parameters):
#         """
#         Initializes the Deja Vu 1 game state parser.

#         Args:
#             pyboy: An instance of the PyBoy emulator.
#             parameters: A dictionary of parameters for configuration.
#         """
#         super().__init__(pyboy, parameters=parameters)
#         events_location = parameters["deja_vu_1_rom_data_path"] + "/events.json"
#         with open(events_location) as f:
#             event_slots = json.load(f)
#         event_slots = event_slots
#         event_names = {v: k for k, v in event_slots.items() if not v[0].isdigit()}
#         beat_opponent_events = bidict()

#         def _pop(d, keys):
#             for key in keys:
#                 if key in d:
#                     d.pop(key, None)

#         pop_queue = []
#         for name, slot in event_names.items():
#             if name.startswith("Beat "):
#                 beat_opponent_events[name.replace("Beat ", "")] = slot
#                 pop_queue.append(name)
#         _pop(event_names, pop_queue)
#         self.defeated_opponent_events = beat_opponent_events
#         """Events related to beating specific opponents. E.g. Beat Brock"""
#         tms_obtained_events = bidict()
#         pop_queue = []
#         for name, slot in event_names.items():
#             if name.startswith("Got Tm"):
#                 tms_obtained_events[name.replace("Got ", "").strip()] = slot
#                 pop_queue.append(name)
#         _pop(event_names, pop_queue)
#         self.tms_obtained_events = tms_obtained_events
#         """Events related to obtaining specific TMs. E.g. Got Tm01"""
#         hm_obtained_events = bidict()
#         pop_queue = []
#         for name, slot in event_names.items():
#             if name.startswith("Got Hm"):
#                 hm_obtained_events[name.replace("Got ", "").strip()] = slot
#                 pop_queue.append(name)
#         _pop(event_names, pop_queue)
#         self.hm_obtained_events = hm_obtained_events
#         """Events related to obtaining specific HMs. E.g. Got Hm01"""
#         passed_badge_check_events = bidict()
#         pop_queue = []
#         for name, slot in event_names.items():
#             if name.startswith("Passed ") and "badge" in name:
#                 passed_badge_check_events[
#                     name.replace("Passed ", "").replace(" Check", "").strip()
#                 ] = slot
#                 pop_queue.append(name)
#         _pop(event_names, pop_queue)
#         self.passed_badge_check_events = passed_badge_check_events
#         """Events related to passing badge checks. E.g. Passed Boulder badge check. These will only be relevant to enter Victory Road."""
#         self.key_items_obtained_events = bidict()
#         """Events related to obtaining key items. E.g. Got Bicycle"""
#         pop_queue = []
#         for name, slot in event_names.items():
#             if name.startswith("Got "):
#                 self.key_items_obtained_events[name.replace("Got ", "").strip()] = slot
#                 pop_queue.append(name)
#         _pop(event_names, pop_queue)
#         self.map_events = {
#             "Cinnabar Gym": bidict(),
#             "Victory Road": bidict(),
#             "Silph Co": bidict(),
#             "Seafoam Islands": bidict(),
#         }
#         """Events related to specific map events like unlocking gates or moving boulders."""
#         for name, slot in event_names.items():
#             if name.startswith("Cinnabar Gym Gate") and name.endswith("Unlocked"):
#                 self.map_events["Cinnabar Gym"][name] = slot
#                 pop_queue.append(name)
#             elif name.startswith("Victory Road") and "Boulder On" in name:
#                 self.map_events["Victory Road"][name] = slot
#                 pop_queue.append(name)
#             elif name.startswith("Silph Co") and "Unlocked" in name:
#                 self.map_events["Silph Co"][name] = slot
#                 pop_queue.append(name)
#             elif name.startswith("Seafoam"):
#                 self.map_events["Seafoam Islands"][name] = slot
#                 pop_queue.append(name)
#         _pop(event_names, pop_queue)
#         self.cutscene_events = bidict()
#         """ Flags for cutscene based events (I think, lol). """

#         cutscenes = [
#             "Event 001",
#             "Daisy Walking",
#             "Pokemon Tower Rival On Left",
#             "Seel Fan Boast",
#             "Pikachu Fan Boast",
#             "Lab Handing Over Fossil Mon",
#             "Route22 Rival Wants Battle",
#         ]  # my best guess, need to verify, Silph Co Receptionist At Desk? Autowalks?
#         pop_queue = []
#         for name, slot in event_names.items():
#             if name in cutscenes:
#                 self.cutscene_events[name] = slot
#                 pop_queue.append(name)
#         _pop(event_names, pop_queue)
#         self.special_events = bidict(event_names)
#         """ All other events not categorized elsewhere."""

#         MAP_PATH = parameters["deja_vu_1_rom_data_path"] + "/map_data.json"
#         with open(MAP_PATH) as map_data:
#             MAP_DATA = json.load(map_data)["regions"]
#         self._MAP_DATA = {int(e["id"]): e for e in MAP_DATA}

#     def get_map_name(self, map_n: int) -> Optional[str]:
#         """
#         Gets the name of the map given its identifier.
#         Args:
#             map_n (int): Map identifier.
#         Returns:
#             Optional[str]: Name of the map if found, None otherwise.
#         """
#         try:
#             return self._MAP_DATA[map_n]["name"]
#         except KeyError:
#             return None

#     def local_to_global(self, r: int, c: int, map_n: int) -> Tuple[int, int]:
#         """
#         Converts local map coordinates to global map coordinates.
#         Args:
#             r (int): Local row coordinate.
#             c (int): Local column coordinate.
#             map_n (int): Map identifier.
#         Returns:
#             (int, int): Global (row, column) coordinates.
#         """
#         try:
#             (
#                 map_x,
#                 map_y,
#             ) = self._MAP_DATA[
#                 map_n
#             ]["coordinates"]
#             gy = r + map_y + self._MAP_ROW_OFFSET
#             gx = c + map_x + self._MAP_COL_OFFSET
#             if (
#                 0 <= gy < self._GLOBAL_MAP_SHAPE[0]
#                 and 0 <= gx < self._GLOBAL_MAP_SHAPE[1]
#             ):
#                 return gy, gx
#             print(
#                 f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})"
#             )
#             return self._GLOBAL_MAP_SHAPE[0] // 2, self._GLOBAL_MAP_SHAPE[1] // 2
#         except KeyError:
#             print(f"Map id {map_n} not found in map_data.json.")
#             return self._GLOBAL_MAP_SHAPE[0] // 2, self._GLOBAL_MAP_SHAPE[1] // 2

#     def get_opponents_defeated(self) -> Set[str]:
#         """
#         Returns a set of all defeated opponents. This function isn't actually used in any current environments, but is left here to show how to read game state information.
#         Similar functions can be created to read obtained TMs, HMs, key items, passed badge checks, etc.

#         Returns:
#             Set[str]: A set of names of defeated opponents.
#         """
#         return self.get_raised_flags(self.defeated_opponent_events)

#     def get_facing_direction(self) -> Tuple[int, int]:
#         """
#         Gets the direction the player is facing.
#         Returns:
#             (int, int): Tuple representing the direction vector (dy, dx).
#         """
#         direction = self.read_m(0xD52A)
#         if direction == 1:
#             return (0, 1)  # Right
#         elif direction == 2:
#             return (0, -1)  # Left
#         elif direction == 4:
#             return (1, 0)  # Down
#         else:
#             return (-1, 0)  # Up

#     def get_local_coords(self) -> Tuple[int, int, int]:
#         """
#         Gets the local game coordinates (x, y, map number).
#         Returns:
#             (int, int, int): Tuple containing (x, y, map number).
#         """
#         return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

#     def get_global_coords(self):
#         """
#         Gets the global coordinates of the player.
#         Returns:
#             (int, int): Tuple containing (global y, global x) coordinates.
#         """
#         x_pos, y_pos, map_n = self.get_local_coords()
#         return self.local_to_global(y_pos, x_pos, map_n)

#     def get_badges(self) -> np.array:
#         """
#         Gets the player's badges as a binary array.
#         Returns:
#             np.array: Array of 8 binary values representing whether the player has obtained each of the badges.
#         """
#         # or  self.bit_count(self.read_m(0xD356))
#         return np.array(
#             [int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8
#         )
