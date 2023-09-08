import datetime
import sys
from typing import List, Optional, Dict, Callable

import numpy as np
import torch
from scipy import stats

# ensure backward compatibility
from baco.param import space
from baco.param.parameters import Parameter, CategoricalParameter, PermutationParameter, OrdinalParameter, RealParameter, IntegerParameter
from baco.param.sampling import random_sample


def get_parameter_neighbors(
        configuration: torch.Tensor,
        parameter: Parameter,
        parameter_idx: int,
        num_neighbors_above: Optional[int] = 3,
        num_neighbors_below: Optional[int] = 3,
) -> torch.Tensor:
    """
    Returns neighbors for a single parameter.

    Input parameters:
        - configuration: The configuration for which to find neighbors to.
        - parameter:
        - param_type:
        - param_idx:
        - num_neighbors_above: The number of neighbors to find with higher values for ordinal parameters.
        - num_neighbors_below: The number of neighbors to find with lower values for ordinal parameters.

    returns:
        - Tensor of neighbors

    For categorical parameters, all neighbors are returned. For permutations all swap-1 neighbors. For ordinal the
    number is defined by the input arguments and for real and integer parameters, the sum of the two num_neighbors
    arguments is used but the distinction between above and below is ignored.

    Does not include the original configuration in neighbors.
    """

    neighbors = []
    if isinstance(parameter, CategoricalParameter):
        for value in parameter.get_discrete_values():
            if configuration[parameter_idx] != value:
                neighbor = configuration.clone()
                neighbor[parameter_idx] = value
                neighbors.append(neighbor)

    elif isinstance(parameter, PermutationParameter):
        if parameter.n_elements > 5:
            for i in range(parameter.n_elements):
                for j in range(i + 1, parameter.n_elements):
                    # swap i and j (probably sloooow)
                    neighbor = configuration.clone()
                    permutation: List[int] = list(parameter.get_permutation_value(configuration[parameter_idx]))
                    j_val = permutation[j]
                    permutation[j] = permutation[i]
                    permutation[i] = j_val

                    neighbor[parameter_idx] = float(
                        parameter.get_int_value(tuple(permutation))
                    )
                    neighbors.append(neighbor)
        else:
            for value in parameter.get_discrete_values():
                if configuration[parameter_idx] != value:
                    neighbor = configuration.clone()
                    neighbor[parameter_idx] = float(value)
                    neighbors.append(neighbor)

    elif isinstance(parameter, OrdinalParameter):
        values = parameter.get_values()
        parameter_value = configuration[parameter_idx]
        value_idx = parameter.val_indices[parameter_value.item()]
        num_neighbors_above = min(num_neighbors_above, len(values) - value_idx)
        num_neighbors_below = min(num_neighbors_below, value_idx)
        values_list = values[value_idx - num_neighbors_below: value_idx] + values[value_idx + 1: value_idx + num_neighbors_above]
        for value in values_list:
            neighbor = configuration.clone()
            neighbor[parameter_idx] = value
            neighbors.append(neighbor)

    elif isinstance(parameter, RealParameter) or isinstance(parameter, IntegerParameter):
        num_neighbors = num_neighbors_below + num_neighbors_above
        mean = parameter.convert(configuration[parameter_idx], "internal", "01")
        a, b = (0 - mean) / 0.1, (1 - mean) / 0.1
        neighboring_values = stats.truncnorm.rvs(a, b, loc=mean, scale=0.1, size=num_neighbors)
        for value in neighboring_values:
            neighbor = configuration.clone()
            neighbor[parameter_idx] = parameter.convert(value, "01", "internal")
            neighbors.append(neighbor)
    else:
        print("Unsupported parameter type")
        raise SystemExit

    if neighbors:
        return torch.cat([n.unsqueeze(0) for n in neighbors], 0)
    else:
        return torch.Tensor()


def get_neighbors(
        configuration: torch.Tensor,
        param_space: space.Space
) -> torch.Tensor:
    """
    Get the neighbors of a configuration

    Input:
        - configuration: dictionary containing the configuration we will generate neighbors for.
        - param_space: a space object containing the search space.

    Returns:
        - a torch.Tensor all the neighbors of 'configuration'
    """

    if param_space.conditional_space:
        return _generate_conditional_neighbors(configuration, param_space)
    else:
        parameters = param_space.parameters
        num_neighbors_above = 3
        num_neighbors_below = 3
        neighbors = configuration.unsqueeze(0)

        for parameter_idx, parameter in enumerate(parameters):
            neighbors = torch.cat(
                (neighbors,
                 get_parameter_neighbors(configuration, parameter, parameter_idx, num_neighbors_above, num_neighbors_below)
                 ), 0)
        return neighbors


def _generate_conditional_neighbors(
        configuration: torch.Tensor,
        param_space: space.Space
) -> torch.Tensor:
    """
    Support method to get_neighbours()
    Input:
        - configuration: configuration for which to find neighbours for
        - param_space: the parameter space object
    Returns:
        - tensor with neighbors
    """

    parameters = param_space.parameters
    num_neighbors_above = 3
    num_neighbors_below = 3
    neighbors = [configuration]

    for parameter_idx, parameter in enumerate(parameters):
        parameter_type = param_space.parameter_types[parameter_idx]

        if parameter_type in ("categorical", "permutation"):
            parameter_neighbors = get_parameter_neighbors(configuration, parameter, parameter_idx, num_neighbors_above, num_neighbors_below)
            feasible = param_space.evaluate(parameter_neighbors, True)
            neighbors.extend([neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f])

        elif parameter_type == "ordinal":
            """
            find feasible above
            """
            feasible_neighbors = []
            tmp_configuration = configuration
            while len(feasible_neighbors) < num_neighbors_above:
                parameter_neighbors = get_parameter_neighbors(tmp_configuration, parameter, parameter_idx, num_neighbors_above - len(feasible_neighbors), 0)
                if len(parameter_neighbors) == 0:
                    break
                feasible = param_space.evaluate(parameter_neighbors, True)
                feasible_neighbors.extend([neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f])
                tmp_configuration = parameter_neighbors[-1]
            neighbors.extend(feasible_neighbors)

            """
            find feasible below
            """
            tmp_configuration = configuration
            while len(feasible_neighbors) < num_neighbors_below:
                parameter_neighbors = get_parameter_neighbors(tmp_configuration, parameter, parameter_idx, 0, num_neighbors_below - len(feasible_neighbors))
                if len(parameter_neighbors) == 0:
                    break
                feasible = param_space.evaluate(parameter_neighbors, True)
                feasible_neighbors.extend([neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f])
                tmp_configuration = parameter_neighbors[-1]
            neighbors.extend(feasible_neighbors)

        elif parameter_type in ("real", "integer"):
            feasible_neighbors = []
            MAX_NUM_ITERATIONS = 10
            num_neighbors = num_neighbors_above + num_neighbors_below
            for iterations in range(MAX_NUM_ITERATIONS):
                if len(feasible_neighbors) >= num_neighbors:
                    break
                parameter_neighbors = get_parameter_neighbors(configuration, parameter, parameter_idx, num_neighbors - len(feasible_neighbors), 0)
                feasible = param_space.evaluate(parameter_neighbors, True)
                feasible_neighbors.extend([neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f])
            neighbors.extend(feasible_neighbors)
    if neighbors:
        return torch.cat([n.unsqueeze(0) for n in neighbors], 0)
    else:
        return torch.Tensor()


def local_search(
        settings: Dict,
        param_space: space.Space,
        acquisition_function: Callable,
        acquisition_function_parameters: Dict,
):
    """
    Optimize the acquisition function using a mix of random and local search.
    This algorithm random samples N points and then does a local search on the
    best points from the random search and the best points from previous iterations (if any).

    Input:
        - settings: BaCO run settings.
        - param_space: a space object containing the search space.
        - acquisition_function: the function that will be optimized by the local search.
        - acquisition_function_parameters: a dictionary containing the parameters that will be passed to the acquisition function.

    Returns:
        - all points evaluated and the best point found by the local search.
    """

    t0 = datetime.datetime.now()
    # percentage of oversampling for the local search starting points
    oversampling_factor = 2

    samples_from_prior = False
    if samples_from_prior:
        random_sample_configurations = random_sample(
            param_space,
            settings["local_search_random_points"],
            "uniform",
            False,
        ) + random_sample(
            param_space,
            settings["local_search_random_points"],
            "using_priors",
            False,
        )
    else:
        random_sample_configurations = random_sample(
            param_space,
            settings["local_search_random_points"],
            "uniform",
            False,
        )

    sampling_time = datetime.datetime.now()
    sys.stdout.write_to_logfile("Total RS time %10.4f sec\n" % ((sampling_time - t0).total_seconds()))

    rs_acquisition_values = acquisition_function(
        settings, param_space, X=random_sample_configurations, **acquisition_function_parameters
    )

    acquisition_time = datetime.datetime.now()
    sys.stdout.write_to_logfile("Optimization function time %10.4f sec\n" % (acquisition_time - sampling_time).total_seconds())

    if settings["local_search_starting_points"] == 0:
        best_index = torch.argmin(rs_acquisition_values).item()
        return random_sample_configurations[best_index]

    best_nbr_of_points = min(settings["local_search_starting_points"] * oversampling_factor, rs_acquisition_values.shape[0])
    best_indices = torch.sort(rs_acquisition_values).indices[:best_nbr_of_points]
    ls_start_configurations = random_sample_configurations[best_indices, :]
    ls_acquisition_values = rs_acquisition_values[best_indices]

    local_minimas = []
    for idx in range(ls_start_configurations.shape[0]):
        configuration = ls_start_configurations[idx]
        scalarization = ls_acquisition_values[idx]

        sys.stdout.write_to_logfile(
            "Starting local search on configuration: "
            + f"<{' '.join(str(x.item()) for x in configuration)}>"
            + f" with acq. val. {scalarization.item()}"
            + "\n"
        )
        while True:
            neighbors = get_neighbors(configuration, param_space)
            neighbor_values = acquisition_function(settings, param_space, X=neighbors, **acquisition_function_parameters)
            if neighbor_values.shape[0] == 0:
                sys.stdout.write_to_logfile("Local minimum found: " + f"<{' '.join(str(x.item()) for x in configuration)}>" + "\n")
                local_minimas.append((configuration, scalarization))
                break

            best_idx = torch.argmin(neighbor_values)
            best_neighbor = neighbors[best_idx]
            best_value = neighbor_values[best_idx]

            sys.stdout.write_to_logfile(
                "Best neighbour: "
                + f"<{' '.join(str(x.item()) for x in best_neighbor)}>"
                + f" with acq. val. {best_value.item()}"
                + "\n"
            )

            if torch.all(torch.eq(configuration, best_neighbor)) or scalarization * (1 - settings["local_search_improvement_threshold"]) <= best_value:
                acquisition_function_parameters["verbose"] = True
                acquisition_function(
                    settings,
                    param_space,
                    X=configuration.unsqueeze(0),
                    **acquisition_function_parameters,
                )
                acquisition_function_parameters["verbose"] = False
                sys.stdout.write_to_logfile("Local minimum found!\n")
                local_minimas.append((configuration, scalarization))
                break
            else:
                configuration = best_neighbor
                scalarization = best_value

    best_idx = np.argmin([s[1] for s in local_minimas])
    best_configuration = local_minimas[best_idx][0]

    local_search_time = datetime.datetime.now()
    sys.stdout.write_to_logfile("Multi-start LS time %10.4f sec\n" % (local_search_time - acquisition_time).total_seconds())

    sys.stdout.write_to_logfile(
        "Best found configuration: "
        + f"<{' '.join(str(x.item()) for x in best_configuration)}>"
        + "\n"
    )

    post_MSLS_time = datetime.datetime.now()

    sys.stdout.write_to_logfile("MSLS time %10.4f sec\n" % (post_MSLS_time - acquisition_time).total_seconds())

    return best_configuration
