##########################################
# NOT USED CURRENTLY
#########################################



import copy
import datetime
import os
import random
import sys
import warnings
from collections import OrderedDict

from jsonschema import exceptions

# ensure backward compatibility
from baco.bo import models
from baco.param import space
from baco.util.util import (
    compute_data_array_scalarization,
    sample_weight_flat,
)
from baco.util.file import (
    add_path,
    set_output_data_file,
)


def main(config, black_box_function=None, profiling=None, bbox_class=None):
    """
    Run design-space exploration using bayesian optimization.
    :param config: dictionary containing all the configuration parameters of this optimization.
    :param output_file: a name for the file used to save the dse results.
    """
    start_time = datetime.datetime.now()
    run_directory = config["run_directory"]
    baco_mode = config["baco_mode"]["mode"]

    # Start logging
    log_file = add_path(config, config["log_file"])
    sys.stdout.change_log_file(log_file)
    if baco_mode == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    # Log the json configuration for this optimization
    sys.stdout.write_to_logfile(str(config) + "\n")

    # Create parameter space object and unpack hyperparameters from json
    param_space = space.Space(config)
    if bbox_class is not None:
        bbox_class.setup(param_space.chain_of_trees, param_space.get_input_parameters())
    application_name = config["application_name"]
    optimization_metrics = config["optimization_objectives"]
    evaluations_per_optimization_iteration = config[
        "evaluations_per_optimization_iteration"
    ]
    set_output_data_file(config, param_space.get_input_output_and_timestamp_parameters())
    batch_mode = evaluations_per_optimization_iteration > 1
    number_of_cpus = config["number_of_cpus"]
    input_params = param_space.get_input_parameters()
    number_of_objectives = len(optimization_metrics)
    objective_limits = {}
    data_array = {}
    fast_addressing_of_data_array = {}
    debug = False

    # Check if some parameters are correctly defined
    if baco_mode == "default":
        if black_box_function == None:
            print("Error: the black box function must be provided")
            raise SystemExit
        if not callable(black_box_function):
            print("Error: the black box function parameter is not callable")
            raise SystemExit

    ### Resume previous optimization, if any
    beginning_of_time = param_space.current_milli_time()
    absolute_configuration_index = 0
    doe_t0 = datetime.datetime.now()
    if config["resume_optimization"] == True:
        resume_data_file = config["resume_optimization_data"]

        if not resume_data_file.endswith(".csv"):
            print("Error: resume data file must be a CSV")
            raise SystemExit
        if resume_data_file == "output_samples.csv":
            resume_data_file = application_name + "_" + resume_data_file

        data_array, fast_addressing_of_data_array = param_space.load_data_file(
            resume_data_file, debug=False, number_of_cpus=number_of_cpus
        )
        absolute_configuration_index = len(
            data_array[list(data_array.keys())[0]]
        )  # get the number of points evaluated in the previous run
        beginning_of_time = (
            beginning_of_time - data_array[param_space.get_timestamp_parameter()[0]][-1]
        )  # Set the timestamp back to match the previous run
        print(
            "Resumed optimization, number of samples = %d ......."
            % absolute_configuration_index
        )

    if data_array:  # if it is not empty
        space.write_data_array(param_space, data_array, output_data_file)

    if param_space.get_conditional_space_flag():
        configurations = param_space.conditional_space_exhaustive_search()
        for configuration in configurations:
            str_data = param_space.get_unique_hash_string_from_values(configuration)
            if str_data in fast_addressing_of_data_array:
                configurations.remove(configuration)
            else:
                fast_addressing_of_data_array[str_data] = 0
        tmp_data_array = param_space.run_configurations(baco_mode, configurations, beginning_of_time, output_data_file, black_box_function, {})
        data_array = concatenate_data_dictionaries(
            data_array,
            tmp_data_array,
            param_space.input_output_and_timestamp_parameter_names,
        )

    print("End of exhaustive search\n")

    sys.stdout.write_to_logfile(
        (
            "Total script time %10.2f sec\n"
            % ((datetime.datetime.now() - start_time).total_seconds())
        )
    )

    return data_array
