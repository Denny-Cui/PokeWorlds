"""
Provides query access to a single VLM that is shared across the project
"""
from poke_worlds.utils.parameter_handling import load_parameters
from poke_worlds.utils.log_handling import log_warn, log_error, log_info
from typing import List, Union
import numpy as np
from PIL import Image
import torch