from poke_worlds.utils import log_info
from poke_worlds.emulation.tracker import MetricGroup, StateTracker, OCRegionMetric, TestTrackerMixin
from poke_worlds.emulation.deja_vu.parsers import (
    DejaVuStateParser,
    AgentState,
    BaseDejaVuStateParser,
    MemoryBasedDejaVuStateParser,
)
from poke_worlds.emulation.deja_vu.base_metrics import (
    DejaVuTestMetric,
    CoreDejaVuMetrics,
)
from poke_worlds.emulation.deja_vu.test_metrics import (
    DejaVuEnterCastleTerminateMetric,
    DejaVuSolveFirstCaseTerminateMetric,
    DejaVuFindFirstClueTerminateMetric,
    DejaVuTalkToCharacterTerminateMetric,
    DejaVuVisitLocationTerminateMetric,
)
from typing import Optional, Type
import numpy as np


class DejaVuOCRMetric(OCRegionMetric):
    REQUIRED_PARSER = DejaVuStateParser

    def reset(self, first=False):
        super().reset(first)
        self.prev_was_in_dialogue = False

    def start(self):
        self.kinds = {
            "dialogue": "dialogue_box_full",
        }
        super().start()

    def can_read_kind(self, current_frame: np.ndarray, kind: str) -> bool:
        self.state_parser: DejaVuStateParser
        if kind == "dialogue":
            # Check if dialogue box is open and not empty
            in_dialogue = self.state_parser.dialogue_box_open(
                current_screen=current_frame
            )
            dialogue_empty = self.state_parser.dialogue_box_empty(
                current_screen=current_frame
            )
            # Check if we're in a menu or puzzle state (not dialogue-specific)
            in_menu = self.state_parser.is_in_menu(
                current_screen=current_frame
            )
            in_puzzle = self.state_parser.is_in_puzzle(
                current_screen=current_frame
            )
            return (
                in_dialogue
                and not dialogue_empty
                and not in_menu
                and not in_puzzle
            )
        return False


class CoreDejaVuTracker(StateTracker):
    """
    StateTracker for core Deja Vu metrics.
    """

    def start(self):
        super().start()
        self.metric_classes.extend([CoreDejaVuMetrics, DejaVuTestMetric])

    def step(self, *args, **kwargs):
        """
        Calls on super().step(), but then modifies the current frame to overlay the grid if the agent is in FREE ROAM.
        """
        super().step(*args, **kwargs)
        state = self.episode_metrics["dejavu_core"]["agent_state"]
        # if agent_state is in FREE ROAM, draw the grid, otherwise do not
        if state == AgentState.FREE_ROAM:
            screen = self.episode_metrics["core"]["current_frame"]
            screen = self.state_parser.draw_grid_overlay(current_frame=screen)
            self.episode_metrics["core"]["current_frame"] = screen
            previous_screens = self.episode_metrics["core"]["passed_frames"]
            if previous_screens is not None:
                self.episode_metrics["core"]["passed_frames"][-1, :] = screen


class DejaVuOCRTracker(CoreDejaVuTracker):
    def start(self):
        super().start()
        self.metric_classes.extend([DejaVuOCRMetric])


class DejaVuTestTracker(TestTrackerMixin, DejaVuOCRTracker):
    """
    Inherit this class and set TERMINATION_TRUNCATION_METRIC to create a TestTracker for Deja Vu games.
    """

    TERMINATION_TRUNCATION_METRIC = None  # Must be set by subclass


class DejaVuEnterCastleTestTracker(DejaVuTestTracker):
    """
    A TestTracker for Deja Vu that ends an episode when the agent enters the castle.
    """

    TERMINATION_TRUNCATION_METRIC = DejaVuEnterCastleTerminateMetric


class DejaVuSolveFirstCaseTestTracker(DejaVuTestTracker):
    """
    A TestTracker for Deja Vu that ends an episode when the agent solves the first case.
    """

    TERMINATION_TRUNCATION_METRIC = DejaVuSolveFirstCaseTerminateMetric


class DejaVuFindFirstClueTestTracker(DejaVuTestTracker):
    """
    A TestTracker for Deja Vu that ends an episode when the agent finds the first clue.
    """

    TERMINATION_TRUNCATION_METRIC = DejaVuFindFirstClueTerminateMetric


class DejaVuTalkToCharacterTestTracker(DejaVuTestTracker):
    """
    A TestTracker for Deja Vu that ends an episode when the agent talks to a specific character.
    """

    TERMINATION_TRUNCATION_METRIC = DejaVuTalkToCharacterTerminateMetric


class DejaVuVisitLocationTestTracker(DejaVuTestTracker):
    """
    A TestTracker for Deja Vu that ends an episode when the agent visits a specific location.
    """

    TERMINATION_TRUNCATION_METRIC = DejaVuVisitLocationTerminateMetric