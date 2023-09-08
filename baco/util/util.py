import copy
import sys
from typing import List, Union, Tuple

import numpy as np
import torch
from scipy import stats

from baco.param.data import DataArray


####################################################
# Data structure handling
####################################################

def are_configurations_equal(configuration1, configuration2):
    """
    Compare two configurations. They are considered equal if they hold the same values for all keys.

    Input:
         - configuration1: the first configuration in the comparison
         - configuration2: the second configuration in the comparison
         - keys: the keys to use for comparison
    Returns:
        - boolean indicating if configurations are equal or not
    """
    for c1, c2 in zip(configuration1, configuration2):
        if c1 != c2:
            return False
    return True


def get_min_configurations(data_array: DataArray, n_points: int) -> DataArray:
    """
    Get the configurations with minimum value according to the comparison key

    Input:
         - configurations: dictionary containing the configurations.
         - number_of_configurations: number of configurations to return.
    Returns:
        - A DataArray wit the best points
    """

    if data_array.metrics_array.shape[1] > 1:
        raise Exception("Trying to call ")

    number_of_configurations = min(n_points, data_array.metrics_array.shape[0])
    best_indices = torch.sort(data_array.metrics_array[0]).indices[:n_points]

    return data_array.slice(best_indices)


def get_min_feasible_configurations(
        data_array: DataArray, n_points: int
):
    """
    Input:
         - data_array: The data among which to select the points
         - n_points: number of configurations to return.
    Returns:
        - a dictionary containing the best configurations.
    """
    feasible_data_array = data_array.get_feasible()
    get_min_configurations(feasible_data_array, n_points)


def lex_sort_unique(matrix: np.ndarray) -> List[bool]:
    """
    checks uniqueness by first sorting the array lexicographically and then comparing neighbors.
    returns a list of bools indicating which indices contain first seen unique values.
    """
    order = np.lexsort(matrix.T)
    matrix = matrix[order]
    diff = np.diff(matrix, axis=0)
    is_unique = np.ones(len(matrix), "bool")
    is_unique[1:] = (diff != 0).any(axis=1)
    return is_unique[order]


def remove_duplicate_configs(
        configurations: Union[np.ndarray, Tuple[np.ndarray]],
        ignore_columns=None,
):
    """
    Removes the duplicates from the combined configurations configs, and lets the first configs keep the remaining
    configurations from the duplicates

    Input:
        - configs: the configurations to be checked for duplicates - duplicates are checked across all configurations, with the first occurrence being kept
        - ignore_column: don't consider the entered columns when checking for duplicates

    Returns:
        - the configurations with duplicates removed
    """
    if isinstance(configurations, tuple):
        merged_configs = np.concatenate(configurations, axis=0)
        config_lengths = [len(c) for c in configurations]
        if ignore_columns is not None:
            merged_configs = np.delete(merged_configs, ignore_columns, axis=1)
        # _, unique_indices = np.unique(merged_configs, return_index=True, axis=0)
        unique_indices = np.arange(len(merged_configs))[
            lex_sort_unique(merged_configs)
        ]

        split_unique_indices = []
        for l in config_lengths:
            split_unique_indices.append([i for i in unique_indices if 0 <= i < l])
            unique_indices -= l
        return [
            configurations[i][split_unique_indices[i]]
            for i in range(len(configurations))
        ]

    else:
        configs_copy = copy.copy(configurations)
        if ignore_columns is not None:
            configs_copy = np.delete(configs_copy, ignore_columns, axis=1)
        # _, unique_indices = np.unique(configs_copy, return_index=True, axis=0)
        unique_indices = np.arange(len(configs_copy))[
            lex_sort_unique(configs_copy)
        ]
        return configurations[unique_indices]


####################################################
# Visualization
####################################################
def get_next_color():
    get_next_color.ccycle = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 0),
        (0, 200, 0),
        (0, 0, 0),
        # get_next_color.ccycle = [(101, 153, 255), (0, 0, 0), (100, 100, 100), (150, 100, 150), (150, 150, 150),
        # (192, 192, 192), (255, 0, 0), (255, 153, 0), (199, 233, 180), (9, 112, 84),
        (0, 128, 0),
        (0, 0, 0),
        (199, 233, 180),
        (9, 112, 84),
        (170, 163, 57),
        (255, 251, 188),
        (230, 224, 123),
        (110, 104, 14),
        (49, 46, 0),
        (138, 162, 54),
        (234, 248, 183),
        (197, 220, 118),
        (84, 105, 14),
        (37, 47, 0),
        (122, 41, 106),
        (213, 157, 202),
        (165, 88, 150),
        (79, 10, 66),
        (35, 0, 29),
        (65, 182, 196),
        (34, 94, 168),
        (12, 44, 132),
        (79, 44, 115),
        (181, 156, 207),
        (122, 89, 156),
        (44, 15, 74),
        (18, 2, 33),
    ]
    get_next_color.color_count += 1
    if get_next_color.color_count > 33:
        return 0, 0, 0
    else:
        a, b, c = get_next_color.ccycle[get_next_color.color_count - 1]
        return float(a) / 255, float(b) / 255, float(c) / 255


get_next_color.color_count = 0


####################################################
# Scalarization
####################################################
def reciprocate_weights(objective_weights):
    """
    Reciprocate weights so that they correlate when using modified_tchebyshev scalarization.

    Input:
         - objective_weights: a dictionary containing the weights for each objective.
    Returns:
        - a dictionary containing the reciprocated weights.
    """
    new_weights = {}
    total_weight = 0
    for objective in objective_weights:
        new_weights[objective] = 1 / objective_weights[objective]
        total_weight += new_weights[objective]

    for objective in new_weights:
        new_weights[objective] = new_weights[objective] / total_weight

    return new_weights


def compute_data_array_scalarization(
        data_array,
        objective_weights,
        objective_means,
        objective_stds,
        scalarization_method,
        log_transform,
        standardize_objectives,
):
    """
    Input:
         - data_array: a dictionary containing the previously run points and their function values.
         - objective_weights: a list containing the weights for each objective.
         - objective_means: a dictionary with estimated (log) mean values for each objective.
         - objective_stds: a dictionary with estimated (log) stds values for each objective.
         - scalarization_method: a string indicating which scalarization method to use.
         - log_transform: bool for if to use log_transformation.
         - log_transform: bool for if to use standardize.
    Returns:
        - a list of scalarized values for each point in data_array and the updated objective limits.
    """
    data_array_len = len(data_array[list(data_array.keys())[0]])

    standardized_data_array = copy.deepcopy(data_array)
    if log_transform:
        for objective in objective_means:
            if min(data_array[objective]) < 0:
                raise Exception(
                    "Cannot use log_transform_outputs for blackbox-functions that take negative values."
                )
            standardized_data_array[objective] = [
                np.log(y) for y in standardized_data_array[objective]
            ]

    if standardize_objectives:
        if not (objective_means is None or objective_stds is None):
            for objective in objective_means:
                standardized_col = [
                    (y - objective_means[objective]) / objective_stds[objective]
                    for y in standardized_data_array[objective]
                ]
                standardized_data_array[objective] = standardized_col
        else:
            sys.stdout.write_to_logfile(
                "Warning: no statistics provided, skipping objective normalization.\n"
            )

    if scalarization_method == "linear":
        scalarized_objectives = np.zeros(data_array_len)
        for run_index in range(data_array_len):
            for objective in objective_weights:
                scalarized_objectives[run_index] += (
                        objective_weights[objective]
                        * standardized_data_array[objective][run_index]
                )

    # The paper does not propose this, we apply their methodology to the original tchebyshev to get the approach below
    # Important: since this was not proposed in the paper, their proofs and bounds for the modified_tchebyshev may not be valid here.
    elif scalarization_method == "tchebyshev":
        scalarized_objectives = np.zeros(data_array_len)
        for run_index in range(data_array_len):
            total_value = 0
            for objective in objective_weights:
                scalarized_value = objective_weights[objective] * abs(
                    standardized_data_array[objective][run_index]
                )
                scalarized_objectives[run_index] = max(
                    scalarized_value, scalarized_objectives[run_index]
                )
                total_value += scalarized_value
            scalarized_objectives[run_index] += 0.05 * total_value
    elif scalarization_method == "modified_tchebyshev":
        scalarized_objectives = np.full((data_array_len), float("inf"))
        reciprocated_weights = reciprocate_weights(objective_weights)
        for run_index in range(data_array_len):
            for objective in objective_weights:
                scalarized_value = reciprocated_weights[objective] * abs(
                    standardized_data_array[objective][run_index]
                )
                scalarized_objectives[run_index] = min(
                    scalarized_value, scalarized_objectives[run_index]
                )
            scalarized_objectives[run_index] = -scalarized_objectives[run_index]
    return scalarized_objectives


def sample_weight_flat(optimization_metrics, evaluations_per_optimization_iteration):
    """
    Sample lambdas for each objective following a dirichlet distribution with alphas equal to 1.
    In practice, this means we sample the weights uniformly from the set of possible weight vectors.
    Input:
         - optimization_metrics: a list containing the optimization objectives.
         - evaluations_per_optimization_iteration: number of weight arrays to sample. Currently not used.
    Returns:
        - a list containing the weight of each objective.
    """
    alphas = np.ones(len(optimization_metrics))
    sampled_weights = stats.dirichlet.rvs(
        alpha=alphas, size=evaluations_per_optimization_iteration
    )

    return torch.tensor(sampled_weights)
