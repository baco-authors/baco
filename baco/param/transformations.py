import copy
import sys
from typing import Dict

from sklearn.preprocessing import OneHotEncoder

from baco.param.data import DataArray
from baco.param.parameters import *
from baco.param.space import Space


def transform_data(
        settings: Dict,
        data_array: DataArray,
        param_space: Space,
        objective_means: torch.Tensor,
        objective_stds: torch.Tensor,
):
    """
    Transform input
    """
    X, parametrization_names = preprocess_parameters_array(data_array.parameters_array, param_space)
    """
    Transform output
    """
    Y = data_array.metrics_array.clone()
    if settings["log_transform_output"]:
        Y = torch.log10(Y)

    if settings["standardize_objectives"]:
        if not (objective_means is None or objective_stds is None):
            Y = (Y - torch.ones(Y.shape) * objective_means) / (torch.ones(Y.shape) * objective_stds)

        else:
            sys.stdout.write_to_logfile(
                "Warning: no statistics provided, skipping objective normalization.\n"
            )
    return X, Y, parametrization_names


def preprocess_parameters_array(
        X: torch.Tensor,
        param_space: Space,
):
    """
    Preprocess a data_array before feeding into a regression/classification model.
    The preprocessing standardize non-categorical inputs (if the flag is set).
    It also transforms categorical variables using one-hot encoding and permutation variables
    according to their chosen parametrization.

    Input:
         - data_array: DataArray containing the X values to transform
         - param_space: parameter space object for the current application.
    Returns:
        - preprocessed data array. The returned data array will contain only the keys in input_params.
    """
    X = copy.deepcopy(X)
    new_X = torch.Tensor()
    new_names = []
    for idx, parameter in enumerate(param_space.parameters):
        if (
                isinstance(parameter, RealParameter) or
                isinstance(parameter, IntegerParameter) or
                isinstance(parameter, OrdinalParameter)
        ):
            new_names.append(parameter.name)
            new_X = torch.cat((new_X, X[:, idx].unsqueeze(1)), dim=1)
            if parameter.transform == "log":
                new_X[:, -1] = torch.log10(new_X[:, -1])
            if param_space.normalize_inputs:
                p_min = parameter.get_min()
                p_max = parameter.get_max()
                if parameter.transform == "log":
                    p_min = np.log10(p_min)
                    p_max = np.log10(p_max)
                new_X[:, -1] = (new_X[:, -1] - p_min) / (p_max - p_min)
        elif isinstance(parameter, CategoricalParameter):
            # Categorical variables are encoded as their index, generate a list of "index labels"
            categories = np.arange(parameter.get_size())
            encoder = OneHotEncoder(categories="auto", sparse=False)
            encoder.fit(categories.reshape(-1, 1))
            x = np.array(X[:, idx]).reshape(-1, 1)
            encoded_x = encoder.transform(x)
            for i in range(encoded_x.shape[1]):
                new_names.append(f"{parameter.name}_{categories[i]}")
            new_X = torch.cat((new_X, torch.tensor(encoded_x)), dim=1)

        elif isinstance(parameter, PermutationParameter):
            # Permutation variables are encoded based on their chosen parametrization
            keys, encoded_x = parameter.parametrize(X[:, idx])
            new_names.extend(keys)
            new_X = torch.cat((new_X, torch.tensor(encoded_x)), dim=1)
    return new_X, new_names
