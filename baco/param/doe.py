from typing import Dict, List, Any, Optional
import numpy as np
from baco.param.space import Space
from baco.param.sampling import random_sample
from baco.param.data import DataArray

def get_doe_sample_configurations(
    param_space: Space,
    data_array: DataArray,
    n_samples: int,
    doe_type: str,
    allow_repetitions: Optional[bool] = False,
):
    """
    Get a list of n_samples configurations with no repetitions and that are not already present in fast_addressing_of_data_array.
    The configurations are sampled following the design of experiments (DOE) in the doe input variable.

    Input:
         - space:
         - data_array: previous points
         - n_samples: the number of unique samples needed.
         - doe_type: type of design of experiments (DOE) chosen.
         - allow_repetitions: allow rpeated configurations.
    Returns:
        - torch.tensor
    """
    configurations = []
    alreadyRunRandom = 0
    configurations_count = 0
    if data_array.parameters_array.shape[0] == 0:
        absolute_configuration_index = 1
    else:
        absolute_configuration_index = data_array.parameters_array.shape[0] + 1

    if doe_type == "random sampling":
        configurations = random_sample(
            param_space,
            n_samples,
            "uniform",
            allow_repetitions,
            data_array.string_dict,
        )
    elif doe_type == "embedding random sampling":
        configurations = random_sample(
            param_space,
            n_samples,
            "embedding",
            allow_repetitions,
            data_array.string_dict,
        )
    else:
        print("Error: design of experiment sampling method not found. Exit.")
        exit(1)

    return configurations
