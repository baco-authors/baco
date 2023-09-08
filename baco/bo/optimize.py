from typing import Dict

import numpy as np
import torch

from baco.bo.acquisition_functions import ei, ucb
from baco.bo.local_search import local_search
from baco.bo.prior_acquisition_functions import ei_bopro, ei_pibo
from baco.param import space
from baco.param.data import DataArray


def optimize_acq(
        settings: Dict,
        param_space: space.Space,
        data_array: DataArray,
        regression_models,
        iteration_number,
        objective_weights,
        objective_means,
        objective_stds,
        best_values,
        classification_model=None,
):
    """
    Run one iteration of bayesian optimization with random scalarizations.

    Input:
        - settings: dictionary containing all the settings of this optimization.
        - data_array: a dictionary containing previously explored points and their function values.
        - param_space: parameter space object for the current application.
        - regression_models: the surrogate models used to evaluate points.
        - iteration_number: the current iteration number.
        - objective_weights: objective weights for multi-objective optimization. Not implemented yet.
        - objective_limits: estimated minimum and maximum limits for each objective.
        - classification_model: feasibility classifier for constrained optimization.

    Returns:
        - best configuration.
    """

    optimization_method = settings["optimization_method"]

    acquisition_function_parameters = {"regression_models": regression_models,
                                       "iteration_number": iteration_number,
                                       "classification_model": classification_model,
                                       "objective_weights": objective_weights,
                                       "objective_means": objective_means,
                                       "objective_stds": objective_stds,
                                       "best_values": best_values,
                                       "feasibility_threshold": (
                                           np.random.choice(settings["feasible_output"]["feasibility_threshold"] * 0.1 * np.arange(11))
                                           if "feasible_output" in settings else 0)
                                       }

    if settings["acquisition_function"] == "EI":
        acquisition_function = ei

    elif settings["acquisition_function"] == "UCB":
        acquisition_function = ucb

    elif settings["acquisition_function"] == "EI_BOPRO":
        acquisition_function = ei_bopro
        acquisition_function_parameters["thresholds"] = (
                                                        torch.tensor([np.quantile(data_array.metrics_array[:, i].numpy(), settings["model_good_quantile"])
                                                                      for i in range(data_array.metrics_array.shape[1])]) - objective_means
                                                        ) / objective_stds

    elif settings["acquisition_function"] == "EI_PIBO":
        acquisition_function = ei_pibo
    else:
        raise Exception("Invalid acquisition function specified.")

    best_configuration = local_search(
        settings,
        param_space,
        acquisition_function,
        acquisition_function_parameters,
    )
    return best_configuration
