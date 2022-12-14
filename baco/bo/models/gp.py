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
from typing import Dict, Any, Union

class GPRegressionModel(GPy.models.GPRegression):

    """
    Class implementing our version of the GPy GP kernel.
    """

    def __init__(
        self,
        settings: Dict[str, Any],
        X: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        input:
            - settings: Run settings
            - X: x training data
            - Y: y trainaing data
        """
        X = X.numpy()
        y = y.reshape(-1, 1).numpy()
        super(GPRegressionModel, self).__init__(
            X,
            y,
            kernel=GPy.kern.Matern52(X.shape[1], ARD=True),
            normalizer=False,
        )

        if settings["normalize_inputs"]:
            if settings["lengthscale_prior"]["name"] == "gamma":
                alpha = float(settings["lengthscale_prior"]["parameters"][0])
                beta = float(settings["lengthscale_prior"]["parameters"][1])
                self.kern.lengthscale.set_prior(
                    GPy.priors.Gamma(alpha, beta)
                )
            elif settings["lengthscale_prior"]["name"] == "lognormal":
                mu = float(settings["lengthscale_prior"]["parameters"][0])
                sigma = float(settings["lengthscale_prior"]["parameters"][1])
                self.kern.lengthscale.set_prior(
                    GPy.priors.LogGaussian(mu, sigma)
                )
            if settings["outputscale_prior"]["name"] == "gamma":
                alpha = float(settings["outputscale_prior"]["parameters"][0])
                beta = float(settings["outputscale_prior"]["parameters"][1])
                self.kern.variance.set_prior(
                    GPy.priors.Gamma(alpha, beta)
                )
            if settings["noise_prior"]["name"] == "gamma":
                alpha = float(settings["noise_prior"]["parameters"][0])
                beta = float(settings["noise_prior"]["parameters"][1])
                self.likelihood.variance.set_prior(
                    GPy.priors.Gamma(alpha, beta)
                )
        if not settings["noise"]:
            self.likelihood.variance = 1e-6
            self.likelihood.fix()

    def set_data(self,
        X,
        y,
    ):
        """
        updates the training data for the GP model.
        Input:
            - X: new x training data
            - X: new y training data
        """
        X = X.numpy()
        y = y.reshape(-1, 1).numpy()
        self.set_XY(X=X, Y=y)


    def fit(
        self,
        settings: Dict[str, Any],
        previous_hyperparameters: Union[Dict[str, Any], None],
    ):
        """
        Fits the model hyperparameters.
        Input:
            - settings:
            - previous_hyperparameters: hyperparameters from previous iterations
        """
        if settings["input_noise"]:
            original_X_train = copy.copy(self.X)
            self.X += 1e-5 * (-1 / 2 + np.random.random(self.X.shape))

        with np.errstate(
            divide="ignore", over="ignore", invalid="ignore"
        ):  # GPy's optimize has uncaught warnings that do not affect performance, suppress them so that they do not propagate to BaCo

            # if the initial lengthscales are small and the input space distances too large,
            # the Gram matrix becomes the identity matrix, and the optimizer fails completely
            while np.min(self.kern.K(self.X)) < 1e-10:
                sys.stdout.write_to_logfile(
                    f"Warning: initial lengthscale too short. Multiplying with 2. Highest similarity: {np.min(self.kern.K(self.X))}\n"
                )
                self.kern.lengthscale = (
                    self.kern.lengthscale * 2
                )
            if settings["multistart_hyperparameter_optimization"]:
                worst_log_likelihood = np.inf
                best_log_likelihood = -np.inf
                best_GP = None

                n_initial_points = settings[
                    "multistart_hyperparameter_optimization_initial_points"
                ]
                n_iterations = settings[
                    "multistart_hyperparameter_optimization_iterations"
                ]

                # gen sample points
                sample_points = [
                    (10**(2*np.random.random(len(self.kern.lengthscale))- 1),
                        10 ** (2 * np.random.random() - 1),
                        10 ** (3 * np.random.random() - 5),
                    )
                    for _ in range(n_initial_points)
                ]

                if settings["reuse_gp_hyperparameters"] and previous_hyperparameters:
                    sample_points.append(
                        (
                            tuple(previous_hyperparameters["lengthscale"]),
                            previous_hyperparameters["variance"],
                            previous_hyperparameters["noise"],
                        )
                    )

                # evaluate sample points
                sample_values = []
                for sample_point in sample_points:
                    try:
                        self.kern.lengthscale = sample_point[0]
                        self.kern.variance = sample_point[1]
                        self.likelihood.variance = sample_point[
                            2
                        ]
                        sample_values.append(
                            self._log_marginal_likelihood
                        )
                    except:
                        sample_values.append(-np.inf)
                best_initial_sample_points = [
                    sample_points[i]
                    for i in np.argpartition(sample_values, -n_iterations)[
                        -n_iterations:
                    ]
                ]
                for sample_point in best_initial_sample_points:
                    try:
                        self.kern.lengthscale = sample_point[0]
                        self.kern.variance = sample_point[1]
                        self.likelihood.variance = sample_point[
                            2
                        ]
                        self.optimize()
                        if (
                            self._log_marginal_likelihood
                            > best_log_likelihood
                        ):
                            best_log_likelihood = self._log_marginal_likelihood
                            best_GP = self.to_dict()
                        if (
                            self._log_marginal_likelihood
                            < worst_log_likelihood
                        ):
                            worst_log_likelihood = self._log_marginal_likelihood
                    except Exception as e:
                        pass
                if best_GP is None:
                    raise Exception(
                        f"Failed to fit the GP hyperparameters in all of the {settings['multistart_hyperparameter_optimization_iterations']} iterations."
                    )
                self = self.from_dict(best_GP)
                sys.stdout.write_to_logfile(
                    f"Best log-likelihood: {best_log_likelihood} Worst log-likelihood: {worst_log_likelihood}\n"
                )
            else:
                self.optimize()  # adding optimizer = 'scg' seems to yield slightly more stable lengthscales

            sys.stdout.write_to_logfile(
                f"lengthscales:\n{self.kern.lengthscale}\n"
            )
            sys.stdout.write_to_logfile(
                f"kernel variance:\n{self.kern.variance}\n"
            )
            sys.stdout.write_to_logfile(
                f"noise variance:\n{self.likelihood.variance}\n"
            )
            try:
                sys.stdout.write_to_logfile(
                    f"{self.kern.K(self.X)[:5, :5]}\n"
                )
            except:
                pass
        if settings["input_noise"]:
            self.set_X(original_X_train)

        hyperparameters = {
            "lengthscale": self.kern.lengthscale,
            "variance": self.kern.variance,
            "noise": self.likelihood.variance,
        }
        return hyperparameters



    def get_mean_and_std(
        self, normalized_data, predict_noiseless, use_var=False
    ):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with a GP model.

        Input:
            - normalized_data: list containing points to predict.
            - predict_noiseless: ignore noise when calculating variance
            - var: whether to compute variance or standard deviation.
        Return:
            - the predicted mean and uncertainty for each point
        """
        if predict_noiseless:
            mean, var = self.predict_noiseless(normalized_data.numpy())
        else:
            mean, var = model.predict(normalized_data)
        mean = mean.flatten()
        var = var.flatten()
        var[var < 10**-11] = 10**-11
        if use_var:
            uncertainty = var
        else:
            uncertainty = np.sqrt(var)

        return torch.tensor(mean), torch.tensor(uncertainty)
