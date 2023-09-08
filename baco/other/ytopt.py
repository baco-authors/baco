import sys
from typing import Any

from numpy import inf
from skopt import Optimizer as SkOptimizer
from skopt.space import Real, Integer, Categorical
from skopt.space import Space as SkoptSpace

import baco.param.space as baco_space
from baco.param.constraints import filter_conditional_values
from baco.param.parameters import RealParameter, IntegerParameter, OrdinalParameter, CategoricalParameter, PermutationParameter
from baco.util.file import initialize_output_data_file
from baco.util.util import *

"""
THE CODE IN THIS FILE IS MORE OR LESS A FUNCTIONAL COPY OF THE DOE PART OF YTOPT (2022-05-01). YTOPT ALSO PROVIDES OTHER FEATURES SUCH AS ASYNCHRONOUS SEARCH AND RUNNING MULTIPLE SETTINGS IN PARALLEL WHICH HAS BEEN REMOVED HERE.
"""

"""Synchronous Model-Based Search.

Arguments of SMBS :
* ``learner``

    * ``RF`` : Random Forest (default)
    * ``ET`` : Extra Trees
    * ``GBRT`` : Gradient Boosting Regression Trees
    * ``DUMMY`` :
    * ``GP`` : Gaussian process

* ``liar-strategy``

    * ``cl_max`` : (default)
    * ``cl_min`` :
    * ``cl_mean`` :

* ``acq-func`` : Acquisition function

    * ``LCB`` :
    * ``EI`` :
    * ``PI`` :
    * ``gp_hedge`` : (default)
"""

N_TRIES = 10
FAIL_VALUE = 20000


class YtOptRunner:
    """
    Runner class for ytopt. Uses ytopt's BO framework which is mostly a wrapper around skopt.
    """

    def __init__(
            self,
            liar_strategy="cl_max",
            set_KAPPA=1.96,
            settings=None,
            black_box_function=None,
            **kwargs,
    ):
        self.settings = settings
        self.baco_space = baco_space.Space(settings)
        self.baco_mode = settings["baco_mode"]["mode"]
        self.beginning_of_time = self.baco_space.current_milli_time()
        self.run_directory = settings["run_directory"]
        self.application_name = settings["application_name"]
        initialize_output_data_file(settings, self.baco_space.all_names)
        self.output_data_file = settings["output_data_file"]
        self.black_box_function = black_box_function
        if self.baco_mode != "client-server" and self.black_box_function is None:
            print(
                "Error, HM-Ytopt must be either run in client-server mode or a black-box function must be provided."
            )
            raise SystemExit
        self.feasible_output_name = None
        self.enable_feasible_predictor = None
        if "feasible_output" in settings:
            feasible_output = settings["feasible_output"]
            self.feasible_output_name = feasible_output["name"]
            self.enable_feasible_predictor = feasible_output[
                "enable_feasible_predictor"
            ]

        self.feasibility_constraints = (
                self.enable_feasible_predictor
                or self.baco_space.conditional_space
        )

        if self.baco_mode == "client-server":
            sys.stdout.switch_log_only_on_file(True)

        self.doe_samples = settings["design_of_experiment"]["number_of_samples"]
        self.optimization_iterations = settings["optimization_iterations"]
        if settings["models"]["model"] == "random_forest":
            learner = "RF"
        elif settings["models"]["model"] == "gaussian_process":
            learner = "GP"
        else:
            raise Exception(f"Invalid model {settings['models']['model']}")

        self.all_results = [[], [], []]
        acq_func = "EI"
        set_SEED = np.random.randint(FAIL_VALUE)

        self.optimizer = Optimizer(space=self.get_space(), learner=learner, acq_func=acq_func, liar_strategy=liar_strategy, set_KAPPA=set_KAPPA, set_SEED=set_SEED, set_NI=1)
        self.max_value = FAIL_VALUE

    def get_space(self):

        """
        Transforms the baco parameter space to a skopt parameter space.
        Skopt does not handle neither ordinals nor permutations and those are hence treated as categoricals.
        """
        params = []

        for parameter in self.baco_space.parameters:
            if isinstance(parameter, RealParameter):
                params.append(
                    Real(
                        parameter.min_value,
                        parameter.max_value,
                        name=parameter.name,
                    )
                )
            elif isinstance(parameter, IntegerParameter):
                params.append(
                    Integer(
                        parameter.min_value,
                        parameter.max_value,
                        name=parameter.name,
                    )
                )
            elif isinstance(parameter, CategoricalParameter):
                params.append(
                    Categorical(
                        parameter.values,
                        name=parameter.name,
                    )
                )
            elif isinstance(parameter, OrdinalParameter):
                params.append(
                    Categorical(
                        parameter.values,
                        name=parameter.name,
                    )
                )
            elif isinstance(parameter, PermutationParameter):
                params.append(
                    Categorical(
                        filter_conditional_values(parameter, parameter.constraints, {}),
                        name=parameter.name,
                    )
                )

        return SkoptSpace(params)

    def run(self, configurations: List[Tuple[Any]]):

        """
        Wrapper to bacos running framework. As ytopt does not consider feasibility, infeasible points are treated as taking the largest value seen so far.
        """
        configurations_array = torch.tensor(configurations)

        # if self.baco_space.parameters[0].name in ["chunk_size", "cs"] and self.baco_space.application_name in ["cpp_taco_SpMM", "cpp_taco_SDDMM"]:
        #    configurations_array[:, 0] = torch.min(configurations_array[:, 0] * configurations_array[:, 5], torch.tensor([512]*configurations_array.shape[0]))

        if self.baco_space.conditional_space:
            base_feasible = self.baco_space.evaluate(configurations_array)
        else:
            base_feasible = [True for _ in configurations]
        feasible = copy.copy(base_feasible)
        feasible_indices = [i for i in range(len(feasible)) if feasible[i]]
        feasible_configurations = configurations_array[feasible_indices, :]

        if len(feasible_indices) == 0:
            sys.stdout.write_to_logfile("all configurations base infeasible\n")
            return [(tuple(configurations[i]), self.max_value) for i in range(len(configurations))], base_feasible, feasible

        feasible_data_array = self.baco_space.run_configurations(
            feasible_configurations,
            self.beginning_of_time,
            self.settings,
            black_box_function=self.black_box_function,
        )

        if self.enable_feasible_predictor:
            true_subindices = []
            false_original_indices = []
            for i, idx in enumerate(feasible_indices):
                if not feasible_data_array.feasible_array[i].item():
                    feasible[idx] = False
                    false_original_indices.append(idx)
                else:
                    true_subindices.append(i)

            feasible_indices = [i for i in feasible_indices if i not in false_original_indices]
            feasible_data_array = feasible_data_array.slice(true_subindices)

        if len(feasible_indices) == 0:
            print("all configurations blackbox infeasible\n")
            return [(tuple(configurations[i]), self.max_value) for i in range(len(configurations))], base_feasible, feasible

        feasible_y_values = feasible_data_array.metrics_array.squeeze(1)

        self.max_value = np.min([self.max_value, torch.max(feasible_y_values).item()])
        y_values = [(feasible_y_values[feasible_indices.index(i)].item() if feasible[i] else self.max_value) for i in range(len(configurations))]

        return [(tuple(configurations[i]), y_values[i]) for i in range(len(y_values))], base_feasible, feasible

    def append_results(self, results, base_feasible, feasible):

        for r, bf, f in zip(results, base_feasible, feasible):
            if bf:
                self.all_results[0].append(r[0])
                self.all_results[1].append(r[1])
                self.all_results[2].append(f)

    @property
    def main(self):

        print("MAX TRIES", N_TRIES)
        print("FAIL_VALUE", FAIL_VALUE)

        base_feasible_solutions_found = 0
        doe_iters = 0
        while base_feasible_solutions_found < self.doe_samples:
            doe_iters += 1
            XX = self.optimizer.ask_initial(n_points=int(self.doe_samples - base_feasible_solutions_found))
            results, base_feasible, feasible = self.run(XX)
            self.optimizer.tell(results)
            if doe_iters >= N_TRIES - 1:
                base_feasible = [True] * len(base_feasible)
            self.append_results(results, base_feasible, feasible)
            base_feasible_solutions_found += sum(base_feasible)

        # MAIN LOOP
        sys.stdout.write_to_logfile("Starting main loop")
        for i in range(self.optimization_iterations):
            print("iteration:", i)
            base_feasible = []
            iter = 0
            while not any(base_feasible):
                iter += 1
                configuration = self.optimizer.ask()
                results, base_feasible, feasible = self.run(configuration)
                if iter >= N_TRIES - 1:
                    base_feasible = [True] * len(base_feasible)
                self.append_results(results, base_feasible, feasible)
                self.optimizer.tell(results)

        n_samples = len(self.all_results[0])
        self.all_results.append([0] * n_samples)
        if self.baco_space.enable_feasible_predictor:
            for i, f in enumerate(self.all_results[2]):
                if not f:
                    self.all_results[1][i] = FAIL_VALUE

        parameters_array = torch.tensor([list(self.all_results[0][i]) for i in range(n_samples)])
        metrics_array = torch.tensor(self.all_results[1]).unsqueeze(1)
        if self.baco_space.enable_feasible_predictor:
            feasible_array = torch.tensor(self.all_results[2])
        else:
            feasible_array = torch.tensor([True for _ in range(len(metrics_array))])
        timestamp_array = torch.tensor(self.all_results[-1])
        data_array = DataArray(parameters_array, metrics_array, timestamp_array, feasible_array)
        baco_space.write_data_array(self.baco_space, data_array, self.output_data_file)

        return data_array


class Optimizer:
    def __init__(
            self,
            space,
            learner,
            acq_func,
            liar_strategy,
            set_KAPPA,
            set_SEED,
            set_NI,
            **kwargs,
    ):
        assert learner in [
            "RF",
            "ET",
            "GBRT",
            "GP",
            "DUMMY",
        ], f"Unknown scikit-optimize base_estimator: {learner}"
        assert liar_strategy in "cl_min cl_mean cl_max".split()

        self.space = space
        self.learner = learner
        self.acq_func = acq_func
        self.liar_strategy = liar_strategy
        self.KAPPA = set_KAPPA
        self.SEED = set_SEED
        self.NI = set_NI

        n_init = inf if learner == "DUMMY" else self.NI  # num_workers

        self._optimizer = SkOptimizer(
            dimensions=self.space.dimensions,
            base_estimator=self.learner,
            acq_optimizer="sampling",
            acq_func=self.acq_func,
            acq_func_kwargs={"kappa": self.KAPPA},
            random_state=self.SEED,
            n_initial_points=n_init,
        )

        self.evals = {}
        self.counter = 0

    def _xy_from_dict(self):
        XX = [x for x in self.evals.keys()]
        YY = [self.evals[x] for x in XX]
        return XX, YY

    def to_dict(self, x: list) -> dict:
        return {p.name: x[i] for i, p in enumerate(self.space)}

    def ask(self):
        return self._optimizer.ask(n_points=1)

    def ask_initial(self, n_points):
        return self._optimizer.ask(n_points=n_points)

    def tell(self, xy_data):
        for x, y in xy_data:
            self.evals[x] = y

        self._optimizer.Xi = []
        self._optimizer.yi = []
        XX, YY = self._xy_from_dict()
        self._optimizer.tell(XX, YY)
