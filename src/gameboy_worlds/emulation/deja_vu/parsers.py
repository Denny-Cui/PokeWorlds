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

from gameboy_worlds.emulation.parser import NamedScreenRegion
from gameboy_worlds.utils import (
    log_warn,
    log_info,
    log_error,
    load_parameters,
    verify_parameters,
)
from gameboy_worlds.emulation.parser import StateParser

from typing import Set, List, Type, Dict, Optional, Tuple
import os
from abc import ABC, abstractmethod
from enum import Enum

from pyboy import PyBoy

import numpy as np
from bidict import bidict


class AgentState(Enum):
    """
    0. FREE_ROAM: The agent is freely roaming the game world.
    1. IN_DIALOGUE: The agent is currently in a dialogue state. (e.g. receiving clues, action feedback)
    2. IN_MENU: The agent is currently in a menu state. (e.g. looking at items)
    """

    FREE_ROAM = 0
    IN_DIALOGUE = 1
    IN_MENU = 2


def _get_proper_regions(
    override_regions: List[Tuple[str, int, int, int, int]],
    base_regions: List[Tuple[str, int, int, int, int]],
) -> List[Tuple[str, int, int, int, int]]:
    """Merges base regions with override regions, giving precedence to override regions."""
    if len(override_regions) == 0:
        return base_regions
    proper_regions = override_regions.copy()
    override_names = [region[0] for region in override_regions]
    for region in base_regions:
        if region[0] not in override_names:
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
    """ 
    List of common named screen regions for Deja Vu game.
    
    Deja Vu uses a primarily text/menu-driven interface. These regions help identify:
    - dialogue_top_left_hook: A hook that appears in the top left after certain events, can be used to determine if certain game mechanics are available.
    - menu_bottom_line: A line that appears at the bottom of the screen when any menu is open, can be used to prevent agent interaction with the UI frame of the emulator.
    """

    COMMON_MULTI_TARGET_REGIONS = [
        ("dialogue_box_area", 0, 74, 160, 55),
        ("menu_box_area", 0, 70, 160, 70),
    ]
    """
    List of common multi-target named screen regions for Deja Vu games.

    Deja Vu has certain regions that can contain multiple important visual cues.
    - dialogue_box_area: The area where dialogue text appears. Can contain multiple targets such as clues
    - menu_box_area: The area where menu options appear. Can contain multiple targets such as items or actions.
    """

    COMMON_MULTI_TARGETS = {
        "dialogue_box_area": [
            "nothing_usual",
        ],
    }
    """
    Common multi-targets for Deja Vu game regions.
    - dialogue_box_area:
        - nothing_usual: Point at useless area.
    """

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
        Initializes the PokemonStateParser.
        Args:
            variant (str): The variant of the Pokemon game.
            pyboy (PyBoy): The PyBoy emulator instance.
            parameters (dict): Configuration parameters for the emulator.
            additional_named_screen_region_details (List[Tuple[str, int, int, int, int]]): Parameters associated with additional named screen regions to include.
            additional_multi_target_named_screen_region_details (List[Tuple[str, int, int, int, int]]): Parameters associated with additional multi-target named screen regions to include.
            override_multi_targets (Dict[str, List[str]]): Dictionary mapping region names to lists of target names for multi-target regions.
        """
        verify_parameters(parameters)
        regions = _get_proper_regions(
            override_regions=additional_named_screen_region_details,
            base_regions=self.COMMON_REGIONS,
        )
        self.variant = variant
        if f"{variant}_rom_data_path" not in parameters:
            log_error(
                f"ROM data path not found for variant: {variant}. Add {variant}_rom_data_path to the config files. See configs/pokemon_red_vars.yaml for an example",
                parameters,
            )
        self.rom_data_path = parameters[f"{variant}_rom_data_path"]
        """ Path to the ROM data directory for the specific Pokemon variant."""
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
        multi_target_regions = _get_proper_regions(
            override_regions=additional_multi_target_named_screen_region_details,
            base_regions=self.COMMON_MULTI_TARGET_REGIONS,
        )
        multi_target_region_names = [region[0] for region in multi_target_regions]
        multi_targets = self.COMMON_MULTI_TARGETS.copy()
        for key in override_multi_targets:
            if key in multi_targets:
                multi_targets[key].extend(override_multi_targets[key])
            else:
                multi_targets[key] = override_multi_targets[key]
        multi_target_provided_region_names = list(multi_targets.keys())
        if not set(multi_target_provided_region_names).issubset(
            set(multi_target_region_names)
        ):
            log_error(
                f"Multi-target regions provided in multi_targets do not match the defined multi-target regions. Provided: {multi_target_provided_region_names}, Defined: {multi_target_region_names}",
                parameters,
            )
        for region_name, x, y, w, h in multi_target_regions:
            region_target_paths = {}
            subdir = captures_dir + f"/{region_name}/"
            for target_name in multi_targets.get(region_name, []):
                region_target_paths[target_name] = os.path.join(subdir, target_name)
            region = NamedScreenRegion(
                region_name,
                x,
                y,
                w,
                h,
                parameters=parameters,
                multi_target_paths=region_target_paths,
            )
            named_screen_regions.append(region)
        super().__init__(pyboy, parameters, named_screen_regions)

    def is_in_menu(self, current_screen: np.ndarray) -> bool:
        """
        Determines if any form of menu is currently open (Case Notes, Evidence, Location, etc).

        Args:
            current_screen (np.ndarray): The current screen frame from the emulator.
            trust_previous (bool): If True, trusts that checks for other states have been done.

        Returns:
            bool: True if a menu is open, False otherwise.
        """
        return self.named_region_matches_target(current_screen, "menu_bottom_line")

    def is_in_dialogue(self, current_screen: np.ndarray) -> bool:
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
        return self.named_region_matches_target(
            current_screen, "dialogue_top_left_hook"
        )

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


class DejaVu1StateParser(DejaVuStateParser):
    """Game state parser for Deja Vu I: The Casebooks of Ace Harding."""

    def __init__(self, pyboy, parameters):
        override_regions = []
        override_multi_target_regions = []
        override_multi_targets = {
            "dialogue_box_area": [
                "took_coat",
            ],
        }

        super().__init__(
            variant="deja_vu_1",
            pyboy=pyboy,
            parameters=parameters,
            additional_named_screen_region_details=override_regions,
            additional_multi_target_named_screen_region_details=override_multi_target_regions,
            override_multi_targets=override_multi_targets,
        )

    def __repr__(self):
        return f"<DejaVuParser(variant={self.variant})>"


class DejaVu2StateParser(DejaVuStateParser):
    """Game state parser for Deja Vu II: The Casebooks of Ace Harding."""

    def __init__(self, pyboy, parameters):
        override_regions = []
        override_multi_target_regions = []
        override_multi_targets = {}

        super().__init__(
            variant="deja_vu_2",
            pyboy=pyboy,
            parameters=parameters,
            additional_named_screen_region_details=override_regions,
            additional_multi_target_named_screen_region_details=override_multi_target_regions,
            override_multi_targets=override_multi_targets,
        )

    def __repr__(self):
        return f"<DejaVuParser(variant={self.variant})>"
