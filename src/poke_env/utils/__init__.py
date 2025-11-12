# These are all the utils functions or classes that you may want to import in your project
from poke_env.utils.parameter_handling import load_parameters
from poke_env.utils.log_handling import log_error, log_info, log_warn, log_dict
from poke_env.utils.fundamental import file_makedir
from pandas import isna

def is_none_str(s):
    if s is None:
        return True
    if isinstance(s, str):
        options = ["none", "null", "nan", ""]
        for option in options:
            if s.lower() == option:
                return True
    return isna(s)
