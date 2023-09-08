import os
import sys

import argparse
from typing import Union, Dict

from baco.bo import bo
from baco.other import evolution
from baco.bo import local_search
from baco.other import exhaustive
from baco.util.util import get_min_configurations, get_min_feasible_configurations
from baco.util.file import read_settings_file
from baco.util.logging import Logger


def optimize(settings_file: Union[str, Dict], black_box_function=None):
    if isinstance(settings_file, str):
        settings = read_settings_file(settings_file)
    elif isinstance(settings_file, dict):
        settings = settings_file
    else:
        raise Exception(f"settings_file must be str or dict, not {type(settings_file)}")

    if not os.path.isdir(settings["run_directory"]):
        os.mkdir(settings["run_directory"])

    # set up logging
    if isinstance(sys.stdout, Logger):
        sys.stdout.change_log_file(settings["log_file"])
    else:
        sys.stdout = Logger(settings["log_file"])
    if settings["baco_mode"]["mode"] == "client-server":
        sys.stdout.switch_log_only_on_file(True)
    for s in settings:
        sys.stdout.write_to_logfile(s + ": " + str(settings[s]) + "\n")

    # run optimization method
    if settings["optimization_method"] in ["bayesian_optimization"]:
        data_array = bo.main(settings, black_box_function=black_box_function)

    elif settings["optimization_method"] in ["local_search", "best_first_local_search"]:
        raise Exception("Local search is currently disabled.")
        data_array = local_search.main(settings, black_box_function=black_box_function)

    elif settings["optimization_method"] == "evolutionary_optimization":
        raise Exception("Evolutionary algorithms is currently disabled.")
        data_array = evolution.main(settings, black_box_function=black_box_function)

    elif settings["optimization_method"] == "exhaustive":
        raise Exception("Exhaustive search is currently disabled.")
        data_array = exhaustive.main(settings, black_box_function=black_box_functio)

    elif settings["optimization_method"] == "opentuner":
        from baco.other import opentuner_shell
        args = opentuner_shell.create_namespace(settings)
        args.settings = settings
        args.black_box_function = black_box_function
        opentuner_shell.OpentunerShell.main(args)

    elif settings["optimization_method"] == "ytopt":
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        from baco.other import ytopt
        ytopt_runner = ytopt.YtOptRunner(settings=settings, black_box_function=black_box_function)
        data_array = ytopt_runner.main

    elif settings["optimization_method"] in ["ytopt_ccs"]:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        from baco.other import ytopt_ccs
        ytopt_runner = ytopt_ccs.YtOptRunner(settings=settings, black_box_function=black_box_function)
        data_array = ytopt_runner.main

    else:
        print("Unrecognized optimization method:", settings["optimization_method"])
        raise SystemExit

    # If mono-objective, compute the best point found
    objectives = settings["optimization_objectives"]
    inputs = list(settings["input_parameters"].keys())
    if len(objectives) == 1 and settings["optimization_method"] not in ["opentuner", "ytopt", "ytopt_cs", "ytopt_ccs"]:
        feasible_output = settings["feasible_output"]
        if feasible_output["enable_feasible_predictor"]:
            feasible_output_name = feasible_output["name"]
            best_point = get_min_feasible_configurations(
                data_array, 1
            )
        else:
            best_point = get_min_configurations(data_array, 1)

    sys.stdout.write_protocol("End of BaCO\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="JSON file containing the run settings")
    args = parser.parse_args()

    if "json_file" in args:
        parameters_file = args.json_file
    else:
        print("Error: exactly one argument needed, the parameters json file.")
        exit(1)

    optimize(parameters_file)


if __name__ == "__main__":
    main()
