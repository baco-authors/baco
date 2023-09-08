import os
import sys

import argparse
from typing import Union, Dict, Callable, Optional
import torch

from baco.util.util import get_min_configurations, get_min_feasible_configurations
from baco.util.file import read_settings_file
from baco.util.logging import Logger
from baco.param.data import DataArray



def optimize(settings_file: Union[str, Dict], black_box_function: Optional[Callable] = None):
    """
    Optimize is the main method of BaCO. It takes a problem to optimize and optimization settings
    in the form of a json file or a dict adn then performs the optimization procedure.

    Input:
        - settings_file: is either a json file name or a dict
        - black_box_function: if the function to optimize is a python callable it is supplied here.
    """
    data_array = None
    if isinstance(settings_file, str):
        settings = read_settings_file(settings_file)
    elif isinstance(settings_file, dict):
        settings = settings_file
    else:
        raise Exception(f"settings_file must be str or dict, not {type(settings_file)}")

    # INITIAL SETUP
    if not os.path.isdir(settings["run_directory"]):
        os.mkdir(settings["run_directory"])

    # set up logging
    if isinstance(sys.stdout, Logger):
        sys.stdout.change_log_file(settings["log_file"])
    else:
        sys.stdout = Logger(settings["log_file"])
    if settings["baco_mode"]["mode"] == "client-server":
        sys.stdout.switch_log_only_on_file(True)

    # print settings
    for s in settings:
        sys.stdout.write_to_logfile(s + ": " + str(settings[s]) + "\n")
    sys.stdout.write_to_logfile("\n")

    # run optimization method
    if settings["optimization_method"] in ["bayesian_optimization"]:
        from baco.bo import bo
        data_array = bo.main(settings, black_box_function=black_box_function)

    elif settings["optimization_method"] == "exhaustive":
        from baco.other import exhaustive
        data_array = exhaustive.main(settings, black_box_function=black_box_function)
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


    elif settings["optimization_method"] in ["ytopt_cs", "ytopt_ccs"]:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        from baco.other import ytopt_ccs
        ytopt_runner = ytopt_ccs.YtOptRunner(settings=settings, black_box_function=black_box_function, ccs=(settings["optimization_method"] == "ytopt_ccs"))
        output = ytopt_runner.main
        data_array = DataArray(
            torch.tensor(output[0]),
            torch.tensor(output[1]).unsqueeze(1),
            torch.tensor(output[2]),
            torch.tensor(output[3]),
        )
    else:
        print("Unrecognized optimization method:", settings["optimization_method"])
        raise SystemExit

    # If mono-objective, compute the best point found
    objectives = settings["optimization_objectives"]
    inputs = list(settings["input_parameters"].keys())
    if len(objectives) == 1 and data_array is not None:
        feasible_output = settings["feasible_output"]
        if feasible_output["enable_feasible_predictor"]:
            feasible_output_name = feasible_output["name"]
            best_point = get_min_feasible_configurations(
                data_array, 1
            )
        else:
            best_point = get_min_configurations(data_array, 1)
        print("Best point found:", best_point)
    sys.stdout.write_protocol("End of HyperMapper\n")


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
