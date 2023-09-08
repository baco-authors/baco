import logging
from types import SimpleNamespace  # to simulate opentuners argparse

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.search.manipulator import FloatParameter, IntegerParameter, EnumParameter

from baco.param.space import Space
from baco.util.file import set_output_data_file
from baco.util.util import *

log = logging.getLogger(__name__)


def create_namespace(settings):
    args = SimpleNamespace()
    args.bail_threshold = 500
    args.database = None
    args.display_frequency = 1000000000
    args.generate_bandit_technique = False
    args.label = None
    args.list_techniques = False
    args.machine_class = None
    args.no_dups = False
    args.par = "brr"
    args.parallel_compile = False
    args.parallelism = 4
    args.pipelining = 0
    args.print_params = False
    args.print_search_space_size = False
    args.quiet = False
    args.results_log = None
    args.results_log_details = None
    args.seed_configuration = []
    args.technique = None

    # Time budget
    if settings["time_budget"] != -1:
        args.stop_after = settings["time_budget"]
    else:
        args.stop_after = float("inf")

    # maximum number of evaluations
    args.test_limit = (
            settings["optimization_iterations"]
            + settings["design_of_experiment"]["number_of_samples"]
    )

    return args


class OpentunerShell(MeasurementInterface):
    def __init__(self, *pargs, **kwargs):
        super(OpentunerShell, self).__init__(*pargs, **kwargs)
        self.param_space = Space(self.args.settings)
        if self.param_space.has_real_parameters:
            raise Exception("The opentuner shell doesn't work with real parameters (due to cot things)")
        self.baco_mode = self.args.settings["baco_mode"]["mode"]
        self.beginning_of_time = self.param_space.current_milli_time()
        self.run_directory = self.args.settings["run_directory"]
        self.number_of_cpus = self.args.settings["number_of_cpus"]
        self.application_name = self.args.settings["application_name"]
        set_output_data_file(
            self.args.settings,
            self.param_space.all_names
        )
        self.output_data_file = self.args.settings["output_data_file"]
        self.black_box_function = self.args.black_box_function
        if self.baco_mode != "client-server" and self.black_box_function is None:
            print(
                "Error, HM-Opentuner must be either run in client-server mode or a black-box function must be provided."
            )
            raise SystemExit
        self.feasible_output_name = None
        self.enable_feasible_predictor = None
        if "feasible_output" in self.args.settings:
            feasible_output = self.args.settings["feasible_output"]
            self.feasible_output_name = feasible_output["name"]
            self.enable_feasible_predictor = feasible_output[
                "enable_feasible_predictor"
            ]

        if self.baco_mode == "client-server":
            sys.stdout.switch_log_only_on_file(True)

        # the order of the parameters in the embedding
        self.chain_of_trees = self.param_space.chain_of_trees
        self.parameter_indices = self.chain_of_trees.cot_order
        # with open(
        #     self.output_data_file,
        #     "w",
        # ) as f:
        #     w = csv.writer(f)
        #     w.writerow(self.param_space.get_input_output_and_timestamp_parameters())

    def seed_configurations(self):
        cfg = self.param_space.get_default_configuration()
        if cfg is None:
            return []
        if not self.param_space.evaluate(cfg)[0]:
            return []
        cfg = cfg[0]
        configuration = {}
        parameter_idx = 0
        for tree in self.chain_of_trees.trees:
            node = tree.root
            while node.children:
                n_children = len(node.children)
                child_index, node = [(i, n) for i, n in enumerate(node.children) if n.value == cfg[self.parameter_indices[parameter_idx]]][0]
                configuration[node.parameter_name] = child_index / n_children
                parameter_idx += 1
        return [configuration]  # opentuner expects a list

    def revert_embedding(self, cfg):
        parameters = self.param_space.parameters
        parameter_idx = 0
        partial_configurations = []
        for tree in self.chain_of_trees.trees:
            node = tree.root
            while node.children:
                child_idx = int(np.floor((1 - 1e-6) * cfg[parameters[parameter_idx].name] * len(node.children)))
                node = node.children[child_idx]
                parameter_idx += 1
            partial_configurations.append(node.get_partial_configuration())
        configuration = self.chain_of_trees.to_original_order(torch.cat(partial_configurations))
        return configuration

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data

        if self.param_space.conditional_space:
            configuration = {}
            configuration = self.revert_embedding(cfg)
            data_array = self.param_space.run_configurations(self.baco_mode, configuration.unsqueeze(0), self.beginning_of_time, self.output_data_file, black_box_function=self.black_box_function, run_directory=self.run_directory)
        else:
            data_array = self.param_space.run_configurations(self.baco_mode, [cfg], self.beginning_of_time, self.output_data_file, black_box_function=self.black_box_function, run_directory=self.run_directory)
        y_value = data_array.metrics_array[0][0].item()
        if (
                self.enable_feasible_predictor
                and not data_array.feasible_array[0].item()
        ):
            return opentuner.resultsdb.models.Result(
                state="ERROR", time=y_value
            )  # if unfeasible, consider an 'ERROR'
        else:
            return opentuner.resultsdb.models.Result(time=y_value)

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        json_parameters = self.args.settings[
            "input_parameters"
        ]  # just for better use/readability
        if self.param_space.conditional_space:
            for param_name in json_parameters:
                manipulator.add_parameter(FloatParameter(param_name, 0, 1))
        else:
            for param_name in json_parameters:
                param_type = json_parameters[param_name]["parameter_type"]
                if param_type == "real":
                    manipulator.add_parameter(
                        FloatParameter(
                            param_name,
                            json_parameters[param_name]["values"][0],
                            json_parameters[param_name]["values"][1],
                        )
                    )
                elif param_type == "integer":
                    manipulator.add_parameter(
                        IntegerParameter(
                            param_name,
                            json_parameters[param_name]["values"][0],
                            json_parameters[param_name]["values"][1],
                        )
                    )
                else:  # Ordinal and categorical
                    string_values = map(str, json_parameters[param_name]["values"])
                    manipulator.add_parameter(EnumParameter(param_name, string_values))

        return manipulator

    def program_name(self):
        return self.application_name

    def save_final_config(self, configuration):
        """
        called at the end of autotuning with the best resultsdb.models.Configuration
        """
        pass
