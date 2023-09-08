import sys
from typing import Any

import cconfigspace as CCS
from numpy import inf
from skopt import Optimizer as SkOptimizer

import baco.param.space as baco_space
from baco.param.constraints import filter_conditional_values
from baco.param.parameters import RealParameter, IntegerParameter, OrdinalParameter, CategoricalParameter, PermutationParameter
from baco.util.file import initialize_output_data_file
from baco.util.util import *
from baco.param.doe import get_doe_sample_configurations

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

# install notes for cconfigspace autogen.sh | mkdir build | cd build | "../configure"/make/ "ln -s /usr/local/lib/libcconfigspace.so.0.0.0 /usr/lib/libcconfigspace.so.0.0.0"


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
        self.baco_space = baco_space.Space(settings)
        self.settings = settings
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
        self.learner = "RF"
        self.cs_space = self.get_space_ccs()

        self.all_results = [[], [], []]
        acq_func = "EI"
        set_SEED = np.random.randint(FAIL_VALUE)

        self.optimizer = Optimizer(
            num_workers=1,
            space=self.cs_space,
            learner=self.learner,
            acq_func=acq_func,
            liar_strategy=liar_strategy,
            set_KAPPA=set_KAPPA,
            set_SEED=set_SEED,
            set_NI=1,
        )
        self.max_value = FAIL_VALUE

    def get_space_ccs(self):

        """
        Transforms the baco parameter space to a skopt parameter space.
        Skopt does not handle neither ordinals nor permutations and those are hence treated as categoricals.
        """
        cs = CCS.ConfigurationSpace()

        for parameter in self.baco_space.parameters:
            if isinstance(parameter, RealParameter):
                cs.add_parameter(CCS.NumericalParameter(
                    name=parameter.name,
                    lower=parameter.min_value,
                    upper=parameter.max_value,
                    default=parameter.default
                ))
            elif isinstance(parameter, IntegerParameter):
                cs.add_parameter(CCS.NumericalParameter(
                    name=parameter.name,
                    lower=parameter.min_value,
                    upper=parameter.max_value,
                    default=parameter.default
                ))

            elif isinstance(parameter, CategoricalParameter):
                cs.add_parameter(CCS.CategoricalParameter(
                    name=parameter.name,
                    values=parameter.values,
                    default_index=(list(parameter.values).index(parameter.default) if parameter.default is not None else None)
                ))
            elif isinstance(parameter, OrdinalParameter):
                cs.add_parameter(CCS.OrdinalParameter(
                    name=parameter.name,
                    values=[v.item() for v in parameter.values],
                    default_index=(list(parameter.values).index(parameter.default) if parameter.default is not None else 0)
                ))
            elif isinstance(parameter, PermutationParameter):
                feasible_values = filter_conditional_values(parameter, parameter.constraints, {})
                cs.add_parameter(CCS.CategoricalParameter(
                    name=parameter.name,
                    values=feasible_values,
                    default_index=(list(feasible_values).index(parameter.default) if parameter.default is not None else None)
                ))

        for parameter in self.baco_space.parameters:
            if not isinstance(parameter, PermutationParameter):
                if parameter.constraints:
                    for cts in parameter.constraints:
                        anti_cts = cts
                        if "==" in cts:
                            anti_cts = anti_cts.replace("==", "!=")
                        elif "!=" in cts:
                            anti_cts = anti_cts.replace("!=", "==")
                        elif "<=" in cts:
                            anti_cts = anti_cts.replace("<=", ">")
                        elif ">=" in cts:
                            anti_cts = anti_cts.replace(">=", "<")
                        elif "<" in cts:
                            anti_cts = anti_cts.replace("<", ">=")
                        elif ">" in cts:
                            anti_cts = anti_cts.replace(">", "<=")
                        if " | " in cts:
                            anti_cts = anti_cts.replace(" | ", " || ")
                        if " & " in cts:
                            anti_cts = anti_cts.replace(" & ", " && ")
                        print("==")
                        print(anti_cts)
                        cs.add_forbidden_clause(anti_cts)

        return cs

    def run(self, configurations: List[Tuple[Any]]):

        """
        Wrapper to bacos running framework. As ytopt does not consider feasibility, infeasible points are treated as taking the largest value seen so far.
        """
        configurations_array = torch.tensor(configurations)

        if self.baco_space.conditional_space:
            base_feasible = self.baco_space.evaluate(configurations_array)
        else:
            base_feasible = [True for c in configurations]

        if not all(base_feasible):
            print(configurations_array)
            print(base_feasible)
            raise Exception(f"Infeasible solution found in ytopt")

        data_array = self.baco_space.run_configurations(configurations_array, self.beginning_of_time, self.settings, self.black_box_function)

        y_values = data_array.metrics_array.squeeze(1)

        if self.enable_feasible_predictor:
            feasible = data_array.feasible_array
            feasible_indices = [i for i in range(configurations_array.shape[0]) if feasible[i]]
            feasible_data_array = data_array.get_feasible()
            feasible_y_values = feasible_data_array.metrics_array.squeeze(1)

            if len(feasible_indices) == 0:
                print("all configurations blackbox infeasible\n")
                return [(tuple(configurations[i]), self.max_value) for i in range(len(configurations))], feasible

            if self.max_value == FAIL_VALUE:
                self.max_value = torch.max(feasible_y_values).item()
            else:
                self.max_value = np.max([self.max_value, torch.max(feasible_y_values).item()])

            y_values = [(y_values[i].item() if feasible[i] else self.max_value) for i in range(len(configurations))]
        else:
            y_values = [y.item() for y in y_values]
            feasible = [True for y in y_values]
        return [(tuple(configurations[i]), y_values[i]) for i in range(len(y_values))], feasible

    def append_results(self, results, feasible):
        for r, f in zip(results, feasible):
            self.all_results[0].append(r[0])
            self.all_results[1].append(r[1])
            self.all_results[2].append(f)

    @property
    def main(self):

        print("FAIL_VALUE", FAIL_VALUE)

        tmp_data_array = DataArray(torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor())
        default_configuration = self.baco_space.get_default_configuration().numpy()
        XX = get_doe_sample_configurations(
            self.baco_space,
            tmp_data_array,
            self.doe_samples - 1,
            "random sampling",
            False,
        ).numpy()
        XX = np.concatenate((default_configuration, XX), axis=0)
        results, feasible = self.run(XX)
        self.optimizer.tell(results)
        self.append_results(results, feasible)

        # MAIN LOOP
        sys.stdout.write_to_logfile("Starting main loop")
        for i in range(self.optimization_iterations):
            print("iteration:", i)

            configuration = self.optimizer.ask()
            results, feasible = self.run(configuration)
            self.append_results(results, feasible)
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

        return self.all_results


class Optimizer:
    def __init__(
            self,
            num_workers: int,
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

        print(self.space)
        self._optimizer = SkOptimizer(
            dimensions=self.space,
            base_estimator=self.learner,
            acq_optimizer="sampling",
            acq_func=self.acq_func,
            acq_func_kwargs={"kappa": self.KAPPA, "n_points": 1},
            random_state=self.SEED,
            n_initial_points=n_init,
        )

        self.evals = {}
        self.counter = 0

    def _xy_from_dict(self):
        XX = list(self.evals.keys())
        YY = [self.evals[x] for x in XX]
        return XX, YY

    def to_dict(self, x: list) -> dict:
        res = {}
        hps = self.space.hyperparameters
        for i in range(len(x)):
            res[hps[i].name] = x[i]
        return res

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

