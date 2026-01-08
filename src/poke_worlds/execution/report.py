from poke_worlds.utils import verify_parameters, log_error
from poke_worlds.emulation import StateTracker
from poke_worlds.interface import HighLevelAction, Environment, History


import numpy as np
from copy import deepcopy


from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class ExecutionReport(ABC):
    """ Holds the report of an execution run. 
    """
    REQUIRED_STATE_TRACKER = StateTracker
    """ The required state tracker class for this execution report (needed to guarantee safety of state_info_to_str). """

    def __init__(self, *, environment: Environment, high_level_goal: str, immediate_task: str, initial_plan: str, visual_context: str, exit_conditions: List[str], parameters: dict):
        verify_parameters(parameters)
        self._parameters = parameters
        self._environment = environment
        if not issubclass(type(environment._emulator.state_tracker), self.REQUIRED_STATE_TRACKER):
            log_error(f"Environment's state tracker {type(environment._emulator.state_tracker)} is not compatible with required {self.REQUIRED_STATE_TRACKER} for this ExecutionReport.", parameters)
        self._history_starting_index = len(environment._history) - 1
        self._history: History = None
        """ The history object from the environment at the start of the execution. Is only set when close() is called. Use get_history to access safely. """
        self.high_level_goal = high_level_goal
        """ The overall high level goal of the execution. """
        self.immediate_task = immediate_task
        """ The immediate task the execution was trying to accomplish. """
        self.exit_conditions = exit_conditions
        """ The exit conditions provided for the execution. """
        self.steps_taken = 0
        """ Number of steps taken in the execution. """
        self.step_contexts: List[Tuple[str, str]] = [(None, visual_context)]
        """ List of tuples containing (difference from previous frame, visual context) at each step. """
        self.plans: List[str] = [initial_plan]
        """ List of plans at each step of the execution. """
        self.exit_reasoning: str = None
        self._action_strings: List[str] = []
        """ List of action strings used during the execution. """
        self._action_messages: List[str] = []
        """ List of action messages received during the execution. """

    def _add_step(self, *, action_string: str, action_messages: str, frame_difference: str, visual_context: str, plan: str):
        """ Adds a step to the execution report. """
        self.steps_taken += 1
        self._action_strings.append(action_string)
        self._action_messages.append(action_messages)
        self.step_contexts.append((frame_difference, visual_context))
        self.plans.append(plan)

    def get_history(self) -> History:
        if self.history is None:
            history = self._environment._history[self._history_starting_index: ]
        else:
            history = deepcopy(self.history)
        return history
 
    def get_observations(self) -> List[Any]:
        """ Returns the list of observation dicts received during the execution. """
        history = self.get_history()
        return history.observations
    
    def get_state_infos(self) -> List[Dict[str, Dict[str, Any]]]:
        """ Returns the list of state info dicts received during the execution. """
        history = self.get_history()
        return history.infos
    
    def get_step_frames(self) -> List[np.ndarray]:
        """ Returns the list of screen frames captured at each step of the execution. """
        history = self.get_history()
        return history.get_step_frames()
    
    def get_transition_frames(self) -> List[np.ndarray]:
        """ Returns the list of transition frames captured between each action execution. """
        history = self.get_history()
        return history.get_transition_frames()
    
    def get_actions_taken(self) -> List[Tuple[str, HighLevelAction, Dict[str, Any], Dict[str, Dict[str, Any]], int, Dict[str, Any], str]]:
        """
        Docstring for get_actions_taken
        
        :param self: Description
        :return: List of actions details taken during the execution. Each entry is a tuple of (action_string, action_class, action_kwargs, transition_states, success_code, action_return_info, action_message).
        :rtype: List[Tuple[str, Any, Dict[str, Any], int, Any, str]]
        """
        history = self.get_history()
        action_details = history.get_action_details()
        use_action_details = []
        for i, action_detail in enumerate(action_details):
            action_class, action_kwargs, transition_states, success_code, action_return_info = action_detail
            action_string = self._action_strings[i]
            action_message = self._action_messages[i]
            use_action_details.append((action_string, action_class, action_kwargs, transition_states, success_code, action_return_info, action_message))
        return use_action_details
    
    def _close(self, exit_reasoning: str):
        """ Closes the execution report with the given exit reasoning. """
        self.exit_reasoning = exit_reasoning
        if self.history is not None:
            log_error("ExecutionReport is already closed.", self._parameters)
        self.history = self.get_history()

    def get_state_info_strings(self) -> List[str]:
        """ Returns the list of state info strings for all state infos in the report. """
        return [self.state_info_to_str(state_info) for state_info in self.get_state_infos()]
        
    @abstractmethod
    def state_info_to_str(self, state_info: dict) -> str:
        """ Converts a state info to a string representation. Useful for VLM Prompting """
        pass
