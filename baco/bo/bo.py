import datetime
import random
import sys

import torch

from baco.bo.models import models
from baco.bo.optimize import optimize_acq
from baco.param import space
from baco.param.data import DataArray
from baco.param.doe import get_doe_sample_configurations
from baco.param.sampling import random_sample
from baco.util.file import load_previous
from baco.util.file import (
    set_output_data_file,
)
from baco.util.settings_check import settings_check_bo
from baco.util.util import (
    sample_weight_flat,
)


def main(settings, black_box_function=None):
    """
    Run design-space exploration using bayesian optimization.

    Input:
        - settings: dictionary containing all the settings of this optimization.
        - black_box_function: a name for the file used to save the dse results.
    """

    ################################################
    # SETUP
    ################################################
    start_time = datetime.datetime.now()

    param_space = space.Space(settings)
    optimization_iterations = settings["optimization_iterations"]
    set_output_data_file(settings, param_space.all_names)
    batch_mode = settings["evaluations_per_iteration"] > 1
    number_of_objectives = len(settings["optimization_objectives"])

    settings = settings_check_bo(settings, black_box_function)

    if "feasible_output" in settings:
        feasible_output = settings["feasible_output"]
        enable_feasible_predictor = feasible_output["enable_feasible_predictor"]
    else:
        enable_feasible_predictor = False
    data_array = DataArray(torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor())

    ################################################
    # RESUME PREVIOUS
    ################################################
    beginning_of_time = param_space.current_milli_time()
    absolute_configuration_index = 0
    doe_t0 = datetime.datetime.now()
    if settings["resume_optimization"]:
        data_array, absolute_configuration_index, beginning_of_time = load_previous(param_space, settings)
        space.write_data_array(param_space, data_array, settings["output_data_file"])

    ################################################
    # DOE
    ################################################
    if absolute_configuration_index < settings["design_of_experiment"]["number_of_samples"]:
        default_configuration = param_space.get_default_configuration()
        doe_parameter_array = torch.Tensor()
        if default_configuration is not None:
            str_data = param_space.get_unique_hash_string_from_values(default_configuration)
            if str_data not in data_array.string_dict:
                doe_parameter_array = default_configuration
                absolute_configuration_index += 1
                data_array.string_dict[str_data] = len(data_array.string_dict.keys())

        if absolute_configuration_index < settings["design_of_experiment"]["number_of_samples"]:
            doe_parameter_array = torch.cat((doe_parameter_array, get_doe_sample_configurations(
                param_space,
                data_array,
                settings["design_of_experiment"]["number_of_samples"] - absolute_configuration_index,
                settings["design_of_experiment"]["doe_type"],
                allow_repetitions=settings["doe_allow_repetitions"],
            )), 0)
            absolute_configuration_index = settings["design_of_experiment"]["number_of_samples"]

        doe_data_array = param_space.run_configurations(settings["baco_mode"]["mode"], doe_parameter_array, beginning_of_time, settings["output_data_file"], black_box_function, settings["run_directory"])

        data_array.cat(doe_data_array)

        iteration_number = 1
    else:
        # if we have more samples than what we require DoE samples we're already in the learning phase
        iteration_number = absolute_configuration_index - settings["design_of_experiment"]["number_of_samples"] + 1

    # If we have feasibility constraints, we must ensure we have at least one feasible sample before starting optimization
    # If this is not true, continue design of experiment until the condition is met
    if enable_feasible_predictor:
        while (
                not True in data_array.feasible_array
        ) and optimization_iterations - iteration_number + 1 > 0:
            print(
                "Warning: all points are invalid, random sampling more configurations."
            )
            print("Number of samples so far:", absolute_configuration_index)
            tmp_parameter_array = get_doe_sample_configurations(
                param_space,
                data_array,
                1,
                "random sampling",
                allow_repetitions=False,
            )
            tmp_data_array = param_space.run_configurations(settings["baco_mode"]["mode"], tmp_parameter_array, beginning_of_time, settings["output_data_file"], black_box_function, settings["run_directory"])
            data_array.cat(tmp_data_array)
            absolute_configuration_index += 1
            iteration_number += 1

        if True not in data_array.feasible_array:
            raise Exception("Budget spent without finding a single feasible solution.")

    feasible_data_array = data_array.get_feasible()

    if settings["log_transform_output"]:
        if torch.min(feasible_data_array.metrics_array) < 0:
            raise Exception("Can't log transform data that take negative values")
        objective_means = torch.mean(torch.log10(feasible_data_array.metrics_array), 0)
        objective_stds = torch.std(torch.log10(feasible_data_array.metrics_array), 0)

    else:
        objective_means = torch.mean(feasible_data_array.metrics_array, 0)
        objective_stds = torch.std(feasible_data_array.metrics_array, 0)

    print(
        "\nEnd of doe/resume phase, the number of evaluated configurations is: %d\n"
        % absolute_configuration_index
    )
    sys.stdout.write_to_logfile(
        (
                "End of DoE - Time %10.4f sec\n"
                % ((datetime.datetime.now() - doe_t0).total_seconds())
        )
    )

    ################################################
    # MAIN LOOP
    ################################################

    # setup
    bo_t0 = datetime.datetime.now()
    if settings["time_budget"] > 0:
        print(
            "starting optimization phase, limited to run for ", settings["time_budget"], " minutes"
        )
    elif settings["time_budget"] == 0:
        print("Time budget cannot be zero. To not limit runtime set time_budget = -1")
        sys.exit()

    model_hyperparameters = None
    # loop
    for iteration in range(iteration_number, optimization_iterations + 1):
        print("Starting optimization iteration", iteration)
        iteration_t0 = datetime.datetime.now()
        #############
        # Get configs
        #############
        # store new configurations until evaluation
        new_parameters_array = torch.Tensor()
        # temporary data array to use until the batch is complete
        tmp_data_array = data_array.copy()
        tmp_feasible_data_array = tmp_data_array.get_feasible()

        for evaluation in range(settings["evaluations_per_iteration"]):
            epsilon = random.uniform(0, 1)
            if epsilon > settings["epsilon_greedy_threshold"]:
                #############
                # Fit models
                #############
                model_t0 = datetime.datetime.now()
                regression_models, model_hyperparameters = models.generate_mono_output_regression_models(
                    settings,
                    tmp_feasible_data_array,
                    param_space,
                    objective_means=objective_means,
                    objective_stds=objective_stds,
                    previous_hyperparameters=model_hyperparameters,
                    reoptimize=(iteration - 1) % settings["reoptimise_hyperparameters_interval"] == 0,
                )
                if regression_models is None:
                    best_configuration = random_sample(
                        param_space,
                        n_samples=1,
                        allow_repetitions=False,
                        previously_run=tmp_data_array.string_dict,
                    ).squeeze(0)
                else:
                    classification_model = None
                    if enable_feasible_predictor and False in data_array.feasible_array:
                        classification_model = models.generate_classification_model(
                            settings,
                            param_space,
                            tmp_data_array,
                        )
                    model_t1 = datetime.datetime.now()
                    sys.stdout.write_to_logfile("Model fitting time %10.4f sec\n" % (model_t1 - model_t0).total_seconds())

                    objective_weights = sample_weight_flat(settings["optimization_objectives"], 1)[0]
                    local_search_t0 = datetime.datetime.now()

                    ##########
                    # optimize
                    ##########
                    best_values = torch.min(feasible_data_array.metrics_array, dim=0)[0]
                    best_configuration = optimize_acq(
                        settings,
                        param_space,
                        tmp_data_array,
                        regression_models,
                        iteration,
                        objective_weights,
                        objective_means,
                        objective_stds,
                        best_values,
                        classification_model,
                    )
                    local_search_t1 = datetime.datetime.now()
                    sys.stdout.write_to_logfile(
                        (
                                "Local search time %10.4f sec\n"
                                % ((local_search_t1 - local_search_t0).total_seconds())
                        )
                    )
            else:
                sys.stdout.write_to_logfile(
                    str(epsilon)
                    + " < "
                    + str(settings["epsilon_greedy_threshold"])
                    + " random sampling a configuration to run\n"
                )
                best_configuration = random_sample(
                    param_space,
                    n_samples=1,
                    allow_repetitions=False,
                    previously_run=tmp_data_array.string_dict,
                ).squeeze(0)

            new_parameters_array = torch.cat((new_parameters_array, best_configuration.unsqueeze(0)), 0)
            # if batch mode, fantasize an objective value
            # if evaluation < settings["evaluations_per_iteration"] - 1:
            #     prediction_means, _ = models.compute_model_mean_and_uncertainty(best_configuration, regression_models, param_space)
            #     tmp_metrics = prediction_means
            #
            #     if classification_model is not None:
            #         classification_prediction_results = classification_model.feas_probability(best_configuration)
            #         true_value_index = (classification_model.classes_.tolist().index(True))
            #         feasibility_indicator = classification_prediction_results[:, true_value_index]
            #         tmp_feasible = torch.tensor([True if feasibility_indicator[0] >= 0.5 else False])
            #     else:
            #         tmp_feasible = torch.Tensor()
            #
            #     tmp_data_array.cat(DataArray(best_configuration, tmp_metric, torch.tensor([0]), tmp_feasible))
            #     tmp_feasible_data_array = tmp_data_array.feasible()
            #
            #     absolute_configuration_index += 1
        ##################
        # Evaluate configs
        ##################
        black_box_function_t0 = datetime.datetime.now()
        new_data_array = param_space.run_configurations(settings["baco_mode"]["mode"], new_parameters_array, beginning_of_time, settings["output_data_file"], black_box_function, settings["run_directory"])
        data_array.cat(new_data_array)
        feasible_data_array = data_array.get_feasible()

        black_box_function_t1 = datetime.datetime.now()
        sys.stdout.write_to_logfile(
            (
                    "Black box function time %10.4f sec\n"
                    % ((black_box_function_t1 - black_box_function_t0).total_seconds())
            )
        )

        # If running batch BO, we will have some liars in fast_addressing_of_data, update them with the true value
        # for configuration_idx in range(
        #     len(new_data_array[list(new_data_array.keys())[0]])
        # ):
        #     configuration = get_single_configuration(
        #         new_data_array, configuration_idx
        #     )
        #     str_data = param_space.get_unique_hash_string_from_values(configuration)
        #     if batch_mode:
        #         if str_data in fast_addressing_of_data_array:
        #             absolute_index = fast_addressing_of_data_array[str_data]
        #             for header in configuration:
        #                 data_array[header][absolute_index] = configuration[header]
        #         else:
        #             fast_addressing_of_data_array[
        #                 str_data
        #             ] = absolute_configuration_index
        #             absolute_configuration_index += 1
        #             for header in configuration:
        #                 data_array[header].append(configuration[header])
        #     else:
        #         for header in configuration:
        #             data_array[header].append(configuration[header])

        if settings["log_transform_output"]:
            if torch.min(feasible_data_array.metrics_array) < 0:
                raise Exception("Can't log transform data that take negative values")
            objective_means = torch.mean(torch.log10(feasible_data_array.metrics_array), 0)
            objective_stds = torch.std(torch.log10(feasible_data_array.metrics_array), 0)

        else:
            objective_means = torch.mean(feasible_data_array.metrics_array, 0)
            objective_stds = torch.std(feasible_data_array.metrics_array, 0)

        run_time = (datetime.datetime.now() - start_time).total_seconds() / 60
        iteration_t1 = datetime.datetime.now()
        sys.stdout.write_to_logfile(
            (
                    "Total iteration time %10.4f sec\n"
                    % ((iteration_t1 - iteration_t0).total_seconds())
            )
        )
        if run_time > settings["time_budget"] != -1:
            break

    sys.stdout.write_to_logfile(
        (
                "End of BO phase - Time %10.4f sec\n"
                % ((datetime.datetime.now() - bo_t0).total_seconds())
        )
    )
    print("End of Bayesian Optimization")

    ################################################
    # POST OPTIMIZATION
    ################################################

    print_posterior_best = settings["print_posterior_best"]
    if print_posterior_best:
        if number_of_objectives > 1:
            print(
                "Warning: print_posterior_best is set to true, but application is not mono-objective."
            )
            print(
                "Can only compute best according to posterior for mono-objective applications. Ignoring."
            )
        elif enable_feasible_predictor:
            print(
                "Warning: print_posterior_best is set to true, but application has feasibility constraints."
            )
            print(
                "Cannot compute best according to posterior for applications with feasibility constraints. Ignoring."
            )
        else:
            # Update model with latest data
            regression_models = models.generate_mono_output_regression_models(
                settings,
                data_array,
                param_space,
                objective_means=objective_means,
                objective_stds=objective_stds,
            )

            best_point = models.minimize_posterior_mean(regression_models, settings, param_space, data_array, objective_means, objective_stds)
            keys = ""
            best_point_string = ""
            for key in best_point:
                keys += f"{key},"
                best_point_string += f"{best_point[key]},"
            keys = keys[:-1]
            best_point_string = best_point_string[:-1]

            sys.stdout.write_protocol("Minimum of the posterior mean:\n")
            sys.stdout.write_protocol(f"{keys}\n")
            sys.stdout.write_protocol(f"{best_point_string}\n\n")

    sys.stdout.write_to_logfile(
        (
                "Total script time %10.2f sec\n"
                % ((datetime.datetime.now() - start_time).total_seconds())
        )
    )

    return data_array
