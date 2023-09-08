###############################################################################################################################
# This script implements the Prior-guided Bayesian Optimization method, presented in: https://arxiv.org/abs/1805.12168.       #
###############################################################################################################################
import datetime
import sys
from typing import Dict, List, Any

import numpy as np
import torch

from baco.bo.acquisition_functions import ei
from baco.bo.models import models
from baco.param.sampling import random_sample
from baco.param.space import Space


def get_prior(
        X: torch.Tensor,
        param_space: Space,
) -> torch.Tensor:
    """
    Compute the probability of configurations being good according to the prior.
    Input:
        - X: tensor with configurations for which to calculate probability
        - param_space: Parameter Space object

    Return:
        - tensor with the probability of each configuration being good/optimal according to the prior.

    I removed what seems to be a not working attempt to combine priors with MO.
    """
    probabilities = torch.zeros(X.shape[0], dtype=torch.float64)

    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            probabilities[row] += np.log(param_space.parameters[col].pdf(X[row, col].item()) + 1e-12)

    return torch.exp(probabilities)


def estimate_prior_limits(
        param_space: Space,
        prior_limit_estimation_points: int,
):
    """
    Estimate the limits for the priors provided. Limits are used to normalize the priors, if prior normalization is required.
    Input:
        - param_space: Space object for the optimization problem
        - prior_limit_estimation_points: number of points to sample to estimate the limits
        - objective_weights: Objective weights for multi-objective optimization. Not implemented yet.
    Return:
        - list with the estimated lower and upper limits found for the prior.
    """
    uniform_configurations = (
        random_sample(
            param_space,
            n_samples=prior_limit_estimation_points,
            sampling_method="uniform",
        )
    )
    prior_configurations = (
        random_sample(
            param_space,
            n_samples=prior_limit_estimation_points,
            sampling_method="using_priors",
        )
    )
    configurations = torch.cat((uniform_configurations, prior_configurations))

    prior = get_prior(configurations, param_space)

    return [min(prior), max(prior)]


def compute_probability_from_model(
        prediction_means: torch.Tensor,
        prediction_stds: torch.Tensor,
        objective_weights: torch.Tensor,
        thresholds: torch.Tensor,
        compute_bad: bool = True,
):
    """
    Compute the probability of configurations to be good or bad according to the model.
    Input:
        - model_means: predicted means of the model for each configuration.
        - model_means: predicted std of the model for each configuration.
        - param_space: Space object for the optimization problem.
        - objective_weights: objective weights for multi-objective optimization. Not implemented yet.
        - threshold: threshold on objective values separating good points and bad points.
        - compute_bad: whether to compute the probability of being good or bad.
    Returns:
        - tensor with probabilities that it is good.
    """
    probabilities = torch.ones(prediction_means.shape[0])
    norm = torch.distributions.normal.Normal(0, 1)

    if compute_bad:
        individual_probabilities = 1 - norm.cdf((thresholds - prediction_means) / prediction_stds)
    else:
        individual_probabilities = norm.cdf((thresholds - prediction_means) / prediction_stds)

    for i in range(individual_probabilities.shape[1]):
        probabilities *= individual_probabilities[:, i] ** objective_weights[i]

    return probabilities


def ei_bopro(
        settings: Dict[str, Any],
        param_space: Space,
        X: torch.Tensor,
        regression_models: List[Any],
        objective_weights: torch.Tensor,
        classification_model: Any,
        thresholds: torch.Tensor,
        iteration_number: int,
        **kwargs,
) -> torch.Tensor:
    """
    Compute the EI acquisition function for a list of configurations based on the priors provided by the user and the BO model.
    Input:
        - configurations: list of configurations to compute EI.
        - param_space: Space object for the optimization problem
        - objective_weights: objective weights for multi-objective optimization. Not implemented yet.
        - threshold: threshold that separates configurations into good or bad for the model.
        - iteration_number: current optimization iteration.
        - regression_models: regression models to compute the probability of a configuration being good according to BO's model.
        - classification_model: classification model to compute the probability of feasibility.
    Returns:
        - tensor with acquisition function values
    """

    model_weight = settings["model_weight"]
    prior_floor = settings["prior_floor"]

    if param_space.normalize_priors:
        good_prior_normalization_limits = estimate_prior_limits(param_space, settings["prior_limit_estimation_points"])
    else:
        good_prior_normalization_limits = None

    if classification_model is not None:
        posterior_normalization_limits = [float("inf"), float("-inf")]
    else:
        posterior_normalization_limits = None

    user_prior_t0 = datetime.datetime.now()
    prior_good = get_prior(X, param_space)

    # if we normalize the prior it's no longer a probability. Is this a problem?
    if good_prior_normalization_limits is not None:
        good_prior_normalization_limits[0] = min(good_prior_normalization_limits[0], min(prior_good))
        good_prior_normalization_limits[1] = max(good_prior_normalization_limits[1], max(prior_good))

        if good_prior_normalization_limits[0] == good_prior_normalization_limits[1]:
            prior_good = torch.ones(len(prior_good), dtype=torch.float64)
        else:
            prior_good = torch.tensor([prior_floor + ((1 - prior_floor) * (x - good_prior_normalization_limits[0])) / (good_prior_normalization_limits[1] - good_prior_normalization_limits[0]) for x in prior_good])

    # if we don't normalize the good_prior, for continuous cases prior good can be larger than one.
    # also it will in most of the cases be << 1 which means that prior bad will be close to 1. Is this a problem?
    prior_bad = 1 - prior_good
    prior_bad[prior_bad < prior_floor] = prior_floor

    sys.stdout.write_to_logfile("EI: user prior time %10.4f sec\n" % ((datetime.datetime.now() - user_prior_t0).total_seconds()))

    model_t0 = datetime.datetime.now()
    number_of_predictions = X.shape[0]

    prediction_means, prediction_stds = models.compute_model_mean_and_uncertainty(
        X,
        regression_models,
        param_space,
        var=False,
        predict_noiseless=settings["predict_noiseless"],
    )

    if classification_model is not None:
        feasibility_indicator = classification_model.feas_probability(X)
        feasibility_indicator[feasibility_indicator == 0] = prior_floor
        feasibility_indicator_log = np.log(feasibility_indicator)

        # Normalize the feasibility indicator to 0, 1.
        # why are we normalizing it post-log?
        feasibility_indicator_log = torch.tensor([prior_floor + ((1 - prior_floor) * (x - np.log(prior_floor))) / (np.log(1) - np.log(prior_floor)) for x in feasibility_indicator_log])
    else:
        feasibility_indicator_log = torch.ones(number_of_predictions)

    model_good = compute_probability_from_model(prediction_means, prediction_stds, objective_weights, thresholds, compute_bad=False)

    model_bad = compute_probability_from_model(prediction_means, prediction_stds, objective_weights, thresholds, compute_bad=True)
    sys.stdout.write_to_logfile("EI: model time %10.4f sec\n" % ((datetime.datetime.now() - model_t0).total_seconds()))

    posterior_t0 = datetime.datetime.now()

    with np.errstate(divide="ignore"):
        # I changed from prior * model^(i/w) -> prior^(w/i) * model. Should be the same, right?
        log_posterior_good = (model_weight / iteration_number) * torch.log(prior_good) + torch.log(model_good)
        log_posterior_bad = (model_weight / iteration_number) * torch.log(prior_bad) + torch.log(model_bad)

    good_bad_ratios = log_posterior_good - log_posterior_bad

    # If we have feasibility constraints, normalize good_bad_ratios to 0, 1
    # but by normalizing in logscale we destroy the intuition of the feasibility indicator??
    # normally acq_val(x) = EI(x) * probability_of_feasibility(x). But this breaks if we normalize them in log-scale.
    if posterior_normalization_limits is not None:
        tmp_gbr = good_bad_ratios.clone()

        # Do not consider -inf and +inf when computing the limits
        tmp_gbr[tmp_gbr == float("-inf")] = float("inf")
        posterior_normalization_limits[0] = min(
            posterior_normalization_limits[0], min(tmp_gbr)
        )
        tmp_gbr[tmp_gbr == float("inf")] = float("-inf")
        posterior_normalization_limits[1] = max(
            posterior_normalization_limits[1], max(tmp_gbr)
        )

        # limits will be equal if all values are the same, in this case, just set the prior to 1 everywhere
        if posterior_normalization_limits[0] == posterior_normalization_limits[1]:
            good_bad_ratios = [1] * len(good_bad_ratios)
        else:
            new_gbr = []
            for x in good_bad_ratios:
                new_x = prior_floor + ((1 - prior_floor) * (x - posterior_normalization_limits[0])) / (posterior_normalization_limits[1] - posterior_normalization_limits[0])
                new_gbr.append(new_x)
            good_bad_ratios = new_gbr
        good_bad_ratios = np.array(good_bad_ratios)

    good_bad_ratios = good_bad_ratios + feasibility_indicator_log
    good_bad_ratios = -1 * good_bad_ratios
    good_bad_ratios = good_bad_ratios

    sys.stdout.write_to_logfile("EI: posterior time %10.4f sec\n" % ((datetime.datetime.now() - posterior_t0).total_seconds()))
    sys.stdout.write_to_logfile("EI: total time %10.4f sec\n" % ((datetime.datetime.now() - user_prior_t0).total_seconds()))

    # local search expects the optimized function to return the values and a feasibility indicator
    return good_bad_ratios


def ei_pibo(
        settings: dict,
        param_space: Space,
        X: torch.Tensor,
        objective_weights: List[float],
        iteration_number: int,
        **kwargs,
) -> torch.Tensor:
    """
    Compute a (multi-objective) EI acquisition function on X.

    Input:
        - settings: run settings for BaCO
        - param_space: the Space object
        - X: a list of tuples containing the points to predict and scalarize.
        - objective_weights: a list containing the weights for each objective.
        - kwargs: used to pass all the additional parameters to EI
    Returns:
        - a tensor of scalarized values for each point in X.
    """

    acquisition_function_values = ei(
        settings,
        param_space,
        X,
        objective_weights,
        **kwargs,
    )

    prior = get_prior(X, param_space)
    # pibo_multipliers = (prior + settings["prior_floor"]) ** (settings["model_weight"] / iteration_number)
    pibo_multipliers = (prior + 1e-6) ** (settings["model_weight"] / iteration_number)
    return acquisition_function_values * pibo_multipliers
