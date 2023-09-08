from typing import Dict, Callable, Union


def settings_check_bo(settings: Dict, black_box_function: Union[None, Callable]) -> Dict:
    """
    All the consistency checks for BO.

    Input:
        - settings: The settings dict for the run.
    Returns:
        - updated settings
    """

    if settings["log_transform_output"] and not (
            settings["acquisition_function"] == "EI"
            and settings["scalarization_method"] == "linear"
    ):
        raise Exception(
            "log_transform_output currently only implemented for linear scalarization with EI"
        )

    # Check if some parameters are correctly defined
    if settings["baco_mode"]["mode"] == "default":
        if black_box_function is None:
            print("Error: the black box function must be provided")
            raise SystemExit
        if not callable(black_box_function):
            print("Error: the black box function parameter is not callable")
            raise SystemExit

    if (settings["models"]["model"] == "gaussian_process") and (settings["acquisition_function"] == "TS"):
        print(
            "Error: The TS acquisition function with Gaussian Process models is still under implementation"
        )
        print("Using EI acquisition function instead")
        settings["acquisition_function"] = "EI"

    if settings["number_of_cpus"] > 1:
        print(
            "Warning: baco supports only sequential execution for now. Running on a single cpu."
        )
        settings["number_of_cpus"] = 1

    return settings
