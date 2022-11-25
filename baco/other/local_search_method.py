##########################################
# NOT USED CURRENTLY
#########################################









import copy
import datetime
import sys
from collections import defaultdict
from multiprocessing import Queue, Process, JoinableQueue
import random
from typing import List, Optional, Any, Dict, Callable
from scipy import stats
from threadpoolctl import threadpool_limits
import numpy as np
import torch

# ensure backward compatibility
from baco.param import space
from baco.param.parameters import Parameter
from baco.param.sampling import random_sample
from baco.param.data import DataArray
from baco.util.util import (
    are_configurations_equal,
    compute_data_array_scalarization,
    get_min_configurations,
    get_min_feasible_configurations,
)
from baco.util.file import (
    add_path,
    set_output_data_file,
)

def get_parameter_neighbors(
    configuration: torch.Tensor,
    parameter: Parameter,
    parameter_type: str,
    parameter_idx: int,
    num_neighbors_above: Optional[int] = 2,
    num_neighbors_below: Optional[int] = 2,
):
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
        - List of neighbors

    For categorical parameters, all neighbors are returned. For permutations all swap-1 neighbors. For ordinal the
    number is defined by the input arguments and for real and integer parameters, the sum of the two num_neighbors
    arguments is used but the distinction between above and below is ignored.

    Does not include the original configuration in neighbors.
    """

    neighbors = []
    if parameter_type == "categorical":
        for value in parameter.get_discrete_values():
            if configuration[parameter_idx] != value:
                neighbor = copy.copy(configuration)
                neighbor[parameter_idx] = value
                neighbors.append(neighbor)

    elif parameter_type == "permutation":
        if parameter.n_elements > 5:
            for i in range(parameter.n_elements):
                for j in range(i + 1, parameter.n_elements):
                    # swap i and j (probably sloooow)
                    neighbor = copy.copy(configuration)
                    permutation = list(
                        param_object.get_permutation_value(configuration[parameter_idx])
                    )
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
                    neighbor = copy.copy(configuration)
                    neighbor[parameter_idx] = float(value)
                    neighbors.append(neighbor)

    elif parameter_type == "ordinal":
        values = parameter.get_discrete_values()
        parameter_value = configuration[parameter_idx]
        # value_idx = values.index(parameter_value)
        value_idx = parameter.val_indices[parameter_value.item()]
        num_neighbors_above = min(num_neighbors_above, len(values) - value_idx)
        num_neighbors_below = min(num_neighbors_below, value_idx)
        values_list = (
            values[value_idx - num_neighbors_below : value_idx]
            + values[value_idx + 1 : value_idx + num_neighbors_above]
        )
        for value in values_list:
            neighbor = copy.copy(configuration)
            neighbor[parameter_idx] = value
            neighbors.append(neighbor)

    elif (parameter_type == "real") or (parameter_type == "integer"):
        num_neighbors = num_neighbors_below + num_neighbors_above
        mean = parameter.convert(configuration[parameter_idx], "internal", "01")
        neighboring_values = stats.truncnorm.rvs(
            0, 1, loc=mean, scale=0.2, size=num_neighbors
        )
        neighboring_values = neighboring_values.tolist()
        for value in neighboring_values:
            neighbor = copy.copy(configuration)
            neighbor[parameter_idx] = parameter.convert(np.clip(value, 0, 1), "01", "internal")
            neighbors.append(neighbor)
    else:
        print("Unsupported parameter type:", param_type)
        raise SystemExit
    if neighbors:
        return torch.cat([n.unsqueeze(0) for n in neighbors],0)
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
        parameter_names = param_space.parameter_names
        num_neighbors_above = 2
        num_neighbors_below = 2
        neighbors = configuration.unsqueeze(0)

        for parameter_idx, parameter in enumerate(parameters):
            parameter_type = param_space.parameter_types[parameter_idx]
            neighbors = torch.cat(
                (neighbors,
                get_parameter_neighbors(
                    configuration,
                    parameter,
                    parameter_type,
                    parameter_idx,
                    num_neighbors_above,
                    num_neighbors_below,
                )
            ),0)
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
    parameter_names = param_space.parameter_names
    num_neighbors_above = 2
    num_neighbors_below = 2
    neighbors = [configuration]

    for parameter_idx, parameter in enumerate(parameters):
        parameter_type = param_space.parameter_types[parameter_idx]

        if parameter_type in ("categorical", "permutation"):
            parameter_neighbors = get_parameter_neighbors(
                configuration,
                parameter,
                parameter_type,
                parameter_idx,
                num_neighbors_above,
                num_neighbors_below,
            )
            feasible = param_space.evaluate(param_neighbors)
            neighbors.extend([neighbor for neighbor, f in zip(param_neighbors, feasible) if f])

        elif parameter_type == "ordinal":
            values = parameter.get_discrete_values()
            parameter_value = configuration[parameter_idx]

            """
            find feasible above
            """
            feasible_neighbors = []
            while (
                len(feasible_neighbors) < num_neighbors_above
            ):
                parameter_neighbors = get_parameter_neighbors(
                    configuration,
                    parameter,
                    parameter_type,
                    parameter_idx,
                    num_neighbors_above - len(feasible_neighbors),
                    0,
                )
                if len(parameter_neighbors) == 0:
                    break
                feasible = param_space.evaluate(parameter_neighbors)
                feasible_neighbors.extend([neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f])
            neighbors.extend(feasible_neighbors)

            """
            find feasible below
            """
            while len(feasible_neighbors) < num_neighbors_below:
                parameter_neighbors = get_parameter_neighbors(
                    configuration,
                    parameter,
                    parameter_type,
                    parameter_idx,
                    0,
                    num_neighbors_below - len(feasible_neighbors),
                )
                if len(parameter_neighbors) == 0:
                    break
                feasible = param_space.evaluate(parameter_neighbors)
                feasible_neighbors.extend([neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f])
            neighbors.extend(feasible_neighbors)

        elif param_type in ("real", "integer"):
            feasible_neighbors = []
            MAX_NUM_ITERATIONS = 10
            num_neighbors = num_neighbors_above + num_neighbors_below
            for iterations in range(MAX_NUM_ITERATIONS):
                if feasible_neighbors >= num_neighbors:
                    break
                parameter_neighbors = get_parameter_neighbors(
                    configuration,
                    parameter,
                    parameter_type,
                    parameter_idx,
                    num_neighbors - len(feasible_neighbors),
                    0,
                )
                feasible = param_space.evaluate(parameter_neighbors)
                feasible_neighbors.extend([neighbor for neighbor, f in zip(parameter_neighbors, feasible) if f])
            neighbors.extend(feasible_neighbors)
    if neighbors:
        return torch.cat([n.unsqueeze(0) for n in neighbors],0)
    else:
        return torch.Tensor()


def run_objective_function(
    settings: Dict,
    param_space: space.Space,
    X: torch.Tensor,
    previous_data_array: DataArray,
    beginning_of_time: int,
    objective_means: torch.Tensor,
    objective_stds: torch.Tensor,
    evaluation_limit=float("inf"),
):
    """
    Evaluate a list of configurations using the black-box function being optimized.

    Input:
        - configurations: list of configurations to evaluate.
        - param_space: a space object containing the search space.
        - beginning_of_time: timestamp of when the optimization started.
        - run_directory: directory where BaCO is running.
        - local_search_data_array: a dictionary containing all of the configurations that have been evaluated.
        - fast_addressing_of_data_array: a dictionary containing evaluated configurations and their index in the local_search_data_array.
        - exhaustive_search_data_array: dictionary containing all points and function values, used in exhaustive mode.
        - exhaustive_search_fast_addressing_of_data_array: dictionary containing the index of each point in the exhaustive array.
        - objective_limits: dictionary containing the estimated min and max limits for each objective.
        - evaluation_limit: the maximum number of function evaluations allowed for the local search.
        - black_box_function: the black_box_function being optimized in the local search.

    Returns:
        - the completed data array
    """

    new_configurations = []
    t0 = datetime.datetime.now()
    absolute_configuration_index = len(previous_data_array.string_dict)

    for configuration in configurations:
        str_data = param_space.get_unique_hash_string_from_values(configuration)
        if not str_data in previous_data_array.string_dict:
            if (
                absolute_configuration_index + number_of_new_evaluations < evaluation_limit
            ):
                new_configurations.append(configuration)
                number_of_new_evaluations += 1


    t1 = datetime.datetime.now()
    if len(new_configurations) > 0:
        new_configurations = torch.stack(new_configurations)
        metrics_array, feasible_array, timestamp_array = param_space.run_configurations(
            settings["baco_mode"]["mode"],
            new_configurations,
            beginning_of_time,
            settings["output_data_file"],
            black_box_function,
            exhaustive_search_data_array,
            exhaustive_search_fast_addressing_of_data_array,
            settings["run_directory"],
        )

    new_data_array = DataArray(new_configurations, metrics_array, timestamp_array, feasible_array)

    sys.stdout.write_to_logfile(
        (
            "Time to run new configurations %10.4f sec\n"
            % ((datetime.datetime.now() - t1).total_seconds())
        )
    )
    sys.stdout.write_to_logfile(
        (
            "Total time to run configurations %10.4f sec\n"
            % ((datetime.datetime.now() - t0).total_seconds())
        )
    )

    return new_data_array


def local_search(
    settings: Dict,
    param_space: space.Space,
    previous_data_array: DataArray,
    optimization_function: Callable,
    optimization_function_parameters: Dict,
    optimization_method=None,
    local_search_as_method=False,
):
    """
    Optimize the acquisition function using a mix of random and local search.
    This algorithm random samples N points and then does a local search on the
    best points from the random search and the best points from previous iterations (if any).

    Input:
        - param_space: a space object containing the search space.
        - fast_addressing_of_data_array: A list containing the points that were already explored.
        - optimization_function: the function that will be optimized by the local search.
        - optimization_function_parameters: a dictionary containing the parameters that will be passed to the optimization function.
        - scalarization_key: the name given to the scalarized values.
        - previous_points: previous points that have already been evaluated.

    Returns:
        - all points evaluted and the best point found by the local search.
    """

    t0 = datetime.datetime.now()
    tmp_string_dict = copy.deepcopy(previous_data_array.string_dict)
    input_params = param_space.parameter_names
    feasible_output_name = param_space.feasible_output_name
    # percentage of oversampling for the local search starting points
    oversampling_factor = 2

    samples_from_prior = False
    if samples_from_prior:
        random_sample_configurations = random_sample(
            param_space,
            settings["local_search_random_points"],
            "uniform",
            settings["doe_allow_repetitions"],
            tmp_string_dict,
        ) + random_sample(
            param_space,
            settings["local_search_random_points"],
            "using_priors",
            settings["doe_allow_repetitions"],
            tmp_string_dict,
        )
    else:
        random_sample_configurations = random_sample(
            param_space,
            settings["local_search_random_points"],
            "uniform",
            settings["doe_allow_repetitions"],
            tmp_string_dict,
        )

    sampling_time = datetime.datetime.now()
    sys.stdout.write_to_logfile(
        ("Total RS time %10.4f sec\n" % ((sampling_time - t0).total_seconds()))
    )

    # The random sampling is so fast so we don't need the parallel version
    rs_acquisition_values = optimization_function(
        settings, param_space, X = random_sample_configurations, **optimization_function_parameters
    )

    acquisition_time = datetime.datetime.now()
    sys.stdout.write_to_logfile(
        (
            "Optimization function time %10.4f sec\n"
            % (acquisition_time - sampling_time).total_seconds()
        )
    )
    best_nbr_of_points = min(settings["local_search_starting_points"] * oversampling_factor, rs_acquisition_values.shape[0])
    best_indices = torch.sort(rs_acquisition_values).indices[:best_nbr_of_points]
    ls_start_configurations = random_sample_configurations[best_indices,:]
    ls_acquisition_values = rs_acquisition_values[best_indices]
    data_collection_time = datetime.datetime.now()

    # def step_away_from_duplicate_points(best_configuration: torch.Tensor):
    #     """
    #     Temporary function to handle duplicate samples. Just local search your way until a new point is found.
    #     Start from the local optima. Then do local search with a taboo list called "visited". As soon as we find
    #     a solution we haven't visted we return it.
    #
    #     Hence, it will take the best neighbour of the already visited local minima. Only if all those have also been visted will it move the the next neighbour.
    #     """
    #
    #     replace = best_configuration in previous_data_array.parameters_array
    #     if not replace:
    #         return best_configuration
    #     visited = []
    #     for iter in range(100):
    #         visited.append(best_configuration)
    #         updated = False
    #         neighbors = get_neighbors(best_configuration, param_space)
    #         new_data_array = optimization_function(
    #             settings, param_space, configurations=neighbors, **optimization_function_parameters
    #         )
    #         sorted_neighbors = [(i, neighbors[i]) for i in np.argsort(function_values)]
    #         for i, neighbor in sorted_neighbors:
    #             if not neighbor in visited:
    #                 if not neighbor in previous_points_list:
    #                     return {
    #                         **{
    #                             param: value
    #                             for param, value in zip(input_params, neighbor)
    #                         },
    #                         **{settings["scalarization_key"]: function_values[i]}
    #                     }
    #                 if not updated:
    #                     best_configuration_list = neighbor
    #                     best_configuration = {
    #                         param: value for param, value in zip(input_params, neighbor)
    #                     }
    #                     updated = True
    #
    #     return False

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
            neighbor_values = optimization_function(
                settings, param_space, X = neighbors, **optimization_function_parameters
            )
            if neighbor_values.shape[0] == 0:
                sys.stdout.write_to_logfile(
                    "Local minimum found: "
                    + f"<{' '.join(str(x.item()) for x in configuration)}>"
                    + "\n"
                )
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

            if all(configuration == best_neighbor) or scalarization <= best_value:
                # if not (settings["allow_duplicate_samples"] or local_search_as_method):
                #     configuration_new = step_away_from_duplicate_points(
                #         configuration
                #     )
                #     if configuration_new != configuration:
                #         logstring += (
                #             "Stepped from: "
                #             + f"{[configuration[param_name] for param_name in param_space.parameter_names]}"
                #             + f" with acq. val. {configuration[settings['scalarization_key']]}"
                #             + " to: "
                #             + f"{[configuration_new[param_name] for param_name in param_space.parameter_names]}"
                #             + f" with acq. val. {configuration_new[settings['scalarization_key']]}"
                #             + "\n"
                #         )
                #         configuration = configuration_new

                if not local_search_as_method:
                    optimization_function_parameters["verbose"] = True
                    optimization_function(
                        settings,
                        param_space,
                        X=configuration.unsqueeze(0),
                        **optimization_function_parameters,
                    )
                    optimization_function_parameters["verbose"] = False
                sys.stdout.write_to_logfile("Local minimum found!\n")
                local_minimas.append((configuration, scalarization))
                break
            else:
                configuration = best_neighbor
                scalarization = best_value

    best_idx = np.argmin([s[1] for s in local_minimas])
    best_configuration = local_minimas[best_idx][0]

    local_search_time = datetime.datetime.now()
    sys.stdout.write_to_logfile(
        (
            "Multi-start LS time %10.4f sec\n"
            % (local_search_time - acquisition_time).total_seconds()
        )
    )

    sys.stdout.write_to_logfile(
        "Best found configuration: "
        + f"<{' '.join(str(x.item()) for x in best_configuration)}>"
        # + f" with acq. val. {result_array[settings['scalarization_key']][best_configuration_idx]}"
        + "\n"
    )
    # if not local_search_as_method:
    #     optimization_function_parameters["verbose"] = True
    #     function_values, feasibility_indicators = optimization_function(
    #         settings, param_space, configurations=[best_configuration], **optimization_function_parameters
    #     )
    #     optimization_function_parameters["verbose"] = False

    post_MSLS_time = datetime.datetime.now()

    sys.stdout.write_to_logfile(
        ("MSLS time %10.4f sec\n" % (post_MSLS_time - acquisition_time).total_seconds())

    return best_configuration


def main(settings, black_box_function=None, profiling=None, bbox_class=None):
    """
    Run design-space exploration using local search.

    Input:
        - settings: dictionary containing all the settings of this design-space exploration.
        - black_box_function:
        - profiling:
        - bbox_class:
    """
    param_space = space.Space(settings)
    if bbox_class is not None:
        bbox_class.setup(param_space.chain_of_trees, param_space.parameter_names)

    run_directory = settings["run_directory"]
    application_name = settings["application_name"]
    if settings["baco_mode"]["mode"] == "default":
        if black_box_function == None:
            print("Error: the black box function must be provided")
            raise SystemExit
        if not callable(black_box_function):
            print("Error: the black box function parameter is not callable")
            raise SystemExit

    output_data_file = set_output_data_file(
        settings, param_space.all_names
    )
    number_of_objectives = len(settings["optimization_objectives"])
    # local search will not produce reasonable output if run in parallel - it is therefore disabled
    settings["number_of_cpus"] = 1
    if settings["local_search_evaluation_limit"] == -1:
        settings["local_search_evaluation_limit"] = float("inf")
    if len(settings["local_search_scalarization_weights"]) < len(settings["optimization_objectives"]):
        print(
            "Error: not enough scalarization weights. Received",
            len(settings["local_search_scalarization_weights"]),
            "expected",
            len(settings["optimization_objectives"]),
        )
        raise SystemExit
    if sum(settings["local_search_scalarization_weights"]) != 1:
        sys.stdout.write_to_logfile("Weights must sum 1. Normalizing weights.\n")
        for idx in range(len(settings["local_search_scalarization_weights"])):
            settings["local_search_scalarization_weights"][idx] = settings["local_search_scalarization_weights"][idx] / sum(
                settings["local_search_scalarization_weights"]
            )
        sys.stdout.write_to_logfile("New weights:" + str(settings["local_search_scalarization_weights"]) + "\n")
    objective_weights = {}

    for idx, objective in enumerate(settings["optimization_objectives"]):
        objective_weights[objective] = settings["local_search_scalarization_weights"][idx]

    # warning hack
    settings["local_search_scalarization_weights"] = objective_weights

    exhaustive_search_data_array = None
    exhaustive_search_fast_addressing_of_data_array = None
    if settings["baco_mode"]["mode"] == "exhaustive":
        exhaustive_file = settings["baco_mode"]["exhaustive_search_file"]
        print("Exhaustive mode, loading data from %s ..." % exhaustive_file)
        (
            exhaustive_search_data_array,
            exhaustive_search_fast_addressing_of_data_array,
        ) = param_space.load_data_file(
            exhaustive_file, debug=False, number_of_cpus=settings["number_of_cpus"]
        )

    enable_feasible_predictor = False
    if "feasible_output" in settings:
        feasible_output = settings["feasible_output"]
        feasible_output_name = feasible_output["name"]
        enable_feasible_predictor = feasible_output["enable_feasible_predictor"]
        enable_feasible_predictor_grid_search_on_recall_and_precision = feasible_output[
            "enable_feasible_predictor_grid_search_on_recall_and_precision"
        ]
        feasible_predictor_grid_search_validation_file = feasible_output[
            "feasible_predictor_grid_search_validation_file"
        ]


    debug = False

    log_file = add_path(settings, settings["log_file"])
    sys.stdout.change_log_file(log_file)
    if settings["baco_mode"]["mode"] == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    absolute_configuration_index = 0
    fast_addressing_of_data_array = {}
    local_search_fast_addressing_of_data_array = {}
    local_search_data_array = defaultdict(list)

    beginning_of_time = param_space.current_milli_time()

    optimization_function_parameters = {}
    optimization_function_parameters["beginning_of_time"] = beginning_of_time
    optimization_function_parameters["exhaustive_search_data_array"] = exhaustive_search_data_array
    optimization_function_parameters["exhaustive_search_fast_addressing_of_data_array"] = exhaustive_search_fast_addressing_of_data_array
    optimization_function_parameters["black_box_function"] = black_box_function
    optimization_function_parameters["local_search_data_array"] = local_search_data_array
    optimization_function_parameters["fast_addressing_of_data_array"] = local_search_fast_addressing_of_data_array
    optimization_function_parameters["objective_stds"] = None
    optimization_function_parameters["objective_means"] = None

    print("Starting local search...")
    local_search_t0 = datetime.datetime.now()
    all_samples, best_configuration = local_search(
        settings,
        param_space,
        fast_addressing_of_data_array,
        run_objective_function,
        optimization_function_parameters,
        profiling=profiling,
        optimization_method="local_search",
        local_search_as_method=True,
    )

    print(
        "Local search finished after %d function evaluations"
        % (len(local_search_data_array[settings["optimization_objectives"][0]]))
    )

    print("### End of the local search.")
    return local_search_data_array
