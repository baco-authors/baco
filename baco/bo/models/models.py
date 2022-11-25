import copy
import datetime
import sys
import numpy as np
import warnings
from collections import defaultdict
import GPy
from scipy import stats
import time
import torch
from typing import Any, Tuple, List, Dict

from baco.param.space import Space
from baco.param.data import DataArray
from baco.bo.local_search import local_search
from baco.bo.models.gp import GPRegressionModel
from baco.bo.models.rf import RFRegressionModel, RFClassificationModel
from baco.param.transformations import transform_data, preprocess_parameters_array

def generate_mono_output_regression_models(
    settings: Dict[str, Any],
    data_array: DataArray,
    param_space: Space,
    objective_means: torch.Tensor=None,
    objective_stds: torch.Tensor=None,
    previous_hyperparameters: Dict[str, float]=None,
    reoptimize:bool =True,
) -> Tuple[List[Any], Dict[str, float]]:
    """
    Fit a regression model, supported model types are Random Forest and Gaussian Process.
    This method fits one mono output model for each objective.

    Input:
        - settings: run settings
        - data_array: the data to use for training.
        - param_space: the parameter space
        - objective_means: means for the different objectives. Used for standardization.
        - objective_stds: stds for the different objectives. Used for standardization.
        - previous_hyperparameters: hyperparamters from the last trained GP.
        - reoptimize: if false, the model will just use previous HPs.

    Returns:
        - the regressors
        - hyperparameters
    """
    start_time = datetime.datetime.now()

    X, Y, parametrization_names = transform_data(settings, data_array, param_space, objective_means, objective_stds)
    models = []
    for i, metric in enumerate(param_space.metric_names):
        y = Y[:,i]

        hyperparameters = None
        if settings["models"]["model"] == "gaussian_process":
            model = GPRegressionModel(settings, X, y)
            if reoptimize:
                hyperparameters = model.fit(settings, previous_hyperparameters,)
            else:
                model.kern.lengthscale = tuple(previous_hyperparameters["lengthscale"])
                model.kern.variance = previous_hyperparameters["variance"]
                model.Gaussian_noise.variance = previous_hyperparameters["noise"]

        elif settings["models"]["model"] == "random_forest":
            model = RFRegressionModel(
                n_estimators=settings["models"]["number_of_trees"],
                max_features=settings["models"]["max_features"],
                bootstrap=settings["models"]["bootstrap"],
                min_samples_split=settings["models"]["min_samples_split"],
            )
            model.fit_RF(X, y)

        else:
            print("Unrecognized model type:", RandomForestRegressor)

        models.append(model)
    sys.stdout.write_to_logfile(
        (
            "End of training - Time %10.2f sec\n"
            % ((datetime.datetime.now() - start_time).total_seconds())
        )
    )
    return models, hyperparameters


def generate_classification_model(
    settings,
    param_space,
    data_array,
):
    """
    Fit a Random Forest model (for now it is Random Forest but in the future we will host more models here (e.g. GPs and lattices).

    Input:
        - settings: run settings
        - param_space: parameter space object for the current application.
        - data_array: the data to use for training.
    Returns:
        - the classifier
    """
    start_time = datetime.datetime.now()
    X, names = preprocess_parameters_array(data_array.parameters_array, param_space)
    classifier = RFClassificationModel(
        settings,
        param_space,
        X,
        data_array.feasible_array,
    )

    sys.stdout.write_to_logfile("End of training - Time %10.2f sec\n" % ((datetime.datetime.now() - start_time).total_seconds()))
    return classifier

def compute_model_mean_and_uncertainty(
    data: torch.Tensor,
    models: List[Any],
    param_space: Space,
    var=False,
    predict_noiseless=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points.

    Input:
        - data: tensor containing points to predict.
        - models: models to use for the prediction.
        - param_space: parameter space object for the current application.
        - var: whether to compute variance or standard deviation.
        - predict_noiseless: ignore noise when calculating variance (only GP)

    Returns:
        - the predicted mean and uncertaitny for each point.
    """
    X, names = preprocess_parameters_array(data, param_space)
    means = torch.Tensor()
    uncertainties = torch.Tensor()
    for model in models:
        mean, uncertainty = model.get_mean_and_std(X, predict_noiseless, var)
        means = torch.cat((means, mean.unsqueeze(1)), 1)
        uncertainties = torch.cat((uncertainties, uncertainty.unsqueeze(1)), 1)

    return means, uncertainties


def ls_compute_posterior_mean(configurations, model, model_type, param_space):
    """
    Compute the posterior mean for a list of configurations. This function follows the interface defined by
    BaCo's local search. It receives configurations from the local search and returns their values.

    Input:
        - configurations: configurations to compute posterior mean
        - model: posterior model to use for predictions
        - model_type: string with the type of model being used.
        - param_space: Space object containing the search space.

    Returns:
        - the posterior mean value for each configuration. To satisfy the local search's requirements, also returns a list of feasibility flags, all set to 1.
    """
    configurations = concatenate_list_of_dictionaries(configurations)
    configurations = data_dictionary_to_tuple(configurations, param_space.parameter_names)
    posterior_means, _ = compute_model_mean_and_uncertainty(configurations, model, param_space)

    objective = param_space.metric_names[0]
    return list(posterior_means[objective]), [1] * len(posterior_means[objective])


def minimize_posterior_mean(
    model,
    settings,
    param_space,
    data_array,
    objective_means,
    objective_stds,
    profiling,
):
    """
    Compute the minimum of the posterior model using a multi-start local search.

    Input:
        - model: posterior model to use for predictions
        - settings: the application scenario defined in the json file
        - param_space: Space object containing the search space.
        - data_array: array containing all of the points that have been explored
        - objective_limits: estimated limits for the optimization objective, used to restore predictions to original range.
        - profiling: whether to profile the local search run.

    Returns:
        - the best configuration according to the mean of the posterior model.
    """
    raise Exeception(
        "minimize_posterior_mean() is not up to date. Needs to be looked after."
    )
    local_search_starting_points = settings["local_search_starting_points"]
    local_search_random_points = settings["local_search_random_points"]
    fast_addressing_of_data_array = (
        {}
    )  # We don't mind exploring repeated points in this case
    scalarization_key = settings["scalarization_key"]
    number_of_cpus = settings["number_of_cpus"]
    model_type = settings["models"]["model"]

    optimization_function_parameters = {}
    optimization_function_parameters["model"] = model
    optimization_function_parameters["model_type"] = model_type
    optimization_function_parameters["param_space"] = param_space

    _, best_configuration = local_search(
        local_search_starting_points,
        local_search_random_points,
        param_space,
        fast_addressing_of_data_array,
        False,  # we do not want the local search to consider feasibility constraints
        ls_compute_posterior_mean,
        optimization_function_parameters,
        scalarization_key,
        number_of_cpus,
        previous_points=data_array,
        profiling=profiling,
    )

    objective = param_space.metric_names[0]
    best_configuration[objective] = ls_compute_posterior_mean(
        [best_configuration], model, model_type, param_space
    )[0][0]
    if settings["standardize_objectives"]:
        objective_min, objective_max = (
            objective_limits[objective][0],
            objective_limits[objective][1],
        )
        best_configuration[objective] = (
            best_configuration[objective] * (objective_max - objective_min)
            + objective_min
        )

    return best_configuration
