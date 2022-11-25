from typing import Optional, Dict, List, Any, Union
from baco.param.space import Space
import torch


def random_sample(
    param_space: Space,
    n_samples: Optional[int] = 1,
    sampling_method: Optional[str] = "using_priors",
    allow_repetitions: Optional[bool] = False,
    previously_run: Dict[str,int] = {},
) -> torch.tensor:

    """
    Random samples configurations
    Input:
        - param_space: Space object
        - n_samples: Number of samples to sample
        - sampling_method: "uniform", "using_priors", "embedding sample"
        - allow_repetitions: Allowing sampling duplicate samples. Genereally much faster if allowed.
        - previously_run: tensor with previously run samples.
    Returns:
        - sampled configurations

    If repetitions are disallowed, it also ensures that previously_run_configurations is not among the returned configurations.
    If the space is conditional, it assumes the existence of a chain_of_trees.
    Will return n_samples new samples, not including the ones already in fast_adressig_of_data_array.
    Previously run is a dict with sting-hashes as keys for fast lookup.
    """

    if not param_space.conditional_space:
        return _random_sample_non_constrained(
            param_space,
            n_samples,
            sampling_method,
            allow_repetitions,
            previously_run,
        )

    else:
        return _random_sample_constrained(
            param_space,
            n_samples,
            sampling_method,
            allow_repetitions,
            previously_run,
        )

def _random_sample_non_constrained(
    param_space: Space,
    n_samples: int,
    sampling_method: str,
    allow_repetitions: bool,
    previously_run: Dict[str, int],
) -> torch.Tensor:

    """
    Random samples configurations without constraints
    Input:
        - param_space: Space object
        - n_samples: Number of samples to sample
        - sampling_method: "uniform", "using_priors"
        - allow_repetitions: Allowing sampling duplicate samples. Genereally much faster if allowed.
        - previously_run: tensor with previously run samples.
    Returns:
        - sampled configurations

    If repetitions are disallowed, it also ensures that previously_run_configurations is not among the returned configurations.
    If the space is conditional, it assumes the existence of a chain_of_trees.
    Will return n_samples new samples, not including the ones already in fast_adressig_of_data_array.
    """

    n_previously_run = len(list(previously_run.keys()))
    n_total_samples = n_samples + n_previously_run

    if allow_repetitions or param_space.has_real_parameters:
        parameters = param_space.parameters
        samples = torch.zeros((n_samples, len(parameters)))
        for i, parameter in enumerate(parameters):
            if sampling_method == "using_priors":
                samples[:, i] = parameter.sample(size=n_samples)
            else:
                samples[:, i] = parameter.sample(size=n_samples, uniform=True)
        return samples

    else:
        # depending on the size of te sapce, different approaches are more efficient
        if param_space.size <= n_total_samples:
            return param_space.filter_out_previously_run(param_space.get_space(), previously_run)

        elif param_space.size <= 2 * n_total_samples:
            remaining_space = param_space.filter_out_previously_run(
                param_space.get_space(), previously_run
            )
            remaining_space_size = param_space.size - n_previously_run

            if sampling_method == "uniform":
                probabilities = None

            elif sampling_method == "using_priors":
                # this uses the log-trick to improve numerical stability
                probabilities = np.array([
                    np.exp(np.sum([np.log(param.pdf(x)) for param in param_space.parameters]))
                    for configuration_idx in range(remaining_space_size)
                ])
                probabilities = probabilities / np.sum(probabilities)

            chosen_indices = np.random.choice(remaining_space_size, n_samples, replace=False, p=probabilities)

            return remaining_space[chosen_indices, :]

        else:
            """
            this part can probably be significantly improved.
            """

            use_priors = sampling_method == "using_priors"

            configurations = torch.Tensor()
            parameters = param_space.parameters
            for i in range(30):
                if configurations.shape[0] >= n_samples:
                    break

                potential_configurations = torch.zeros((n_samples - configurations.shape[0], len(parameters)))
                for i, parameter in enumerate(parameters):
                    if sampling_method == "using_priors":
                        potential_configurations[:, i] = parameter.sample(size=n_samples)
                    else:
                        potential_configurations[:, i] = parameter.sample(size=n_samples, uniform=True)

                new_configurations = param_space.filter_out_previously_run(
                    potential_configurations, previously_run
                )
                for c in new_configurations:
                    previously_run[param_space.get_unique_hash_string_from_values(c)] = -1
                configurations = torch.cat((configurations, new_configurations), axis=0)
            return configurations

def _random_sample_constrained(
    param_space: Space,
    n_samples: int,
    sampling_method: str,
    allow_repetitions: bool,
    previously_run_configurations: Dict[str, int],
):
    """
    Random samples constrained configurations
    Input:
        - param_space: Space object
        - n_samples: Number of samples to sample
        - sampling_method: "uniform", "using_priors", "embedding sample"
        - allow_repetitions: Allowing sampling duplicate samples. Genereally much faster if allowed.
        - previously_run: tensor with previously run samples.
    Returns:
        - sampled configurations

    If repetitions are disallowed, it also ensures that previously_run_configurations is not among the returned configurations.
    If the space is conditional, it assumes the existence of a chain_of_trees.
    Will return "n_samples" new samples, not including the ones already in fast_adressig_of_data_array.
    """
    n_previously_run = len(list(previously_run_configurations.keys()))
    n_total_samples = n_samples + n_previously_run

    if allow_repetitions:
        return param_space.chain_of_trees.sample(
            n_samples, sampling_method, param_space.parameter_names, allow_repetitions=True
        )
    else:
        if param_space.chain_of_trees.get_number_of_configurations() <= n_total_samples:
            print(param_space.chain_of_trees.get_all_configurations(), previously_run_configurations,)
            return param_space.filter_out_previously_run(
                param_space.chain_of_trees.get_all_configurations(),
                previously_run_configurations,
            )
        else:
            if not previously_run_configurations:
                return param_space.chain_of_trees.sample(
                    n_samples, sampling_method, param_space.parameter_names, allow_repetitions=False
                )
            else:
                too_many_samples = param_space.chain_of_trees.sample(
                    n_total_samples, sampling_method, allow_repetitions=False
                )
                return param_space.filter_out_previously_run(too_many_samples, previously_run_configurations)[:n_samples]


def get_random_configurations(
    param_space: Space,
    use_priors=True,
    n_samples=1
) -> torch.Tensor:
    """
    Input:
        - param_space: Space object
        - use_priors: whether the prior distributions of the parameters should be used for the sampling
        - n_samples: the number of sampled random points
    Returns:
        - a number of random configurations from the parameter space under the form of a dictionary, or the sampled array, shape (size, dims)
    """

    parameters = param_space.parameters
    samples = torch.zeros((n_samples, len(parameters)))
    for i, parameter in enumerate(parameters):
        if use_priors:
            samples[i, :] = parameter.sample(size=size)
        else:
            samples[i, :] = parameter.sample(size=size, uniform=True)

    return samples
