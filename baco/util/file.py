import os
import csv
import json
from jsonschema import Draft4Validator, validators
from typing import Dict, Tuple, List
import torch
from pkg_resources import resource_stream
from baco.param.space import Space
from baco.param.data import DataArray
#####################################
# JSON
#####################################

def read_settings_file(settings_file):
    """
    Reads a json settings file and returns a settings dict.

    Input:
         - file_name:
    Returns:
        - settings dict
    """
    if not settings_file.endswith(".json"):
        _, file_extension = os.path.splitext(settings_file)
        print(
            "Error: invalid file name. \nThe input file has to be a .json file not a %s"
            % file_extension
        )
        raise SystemExit
    with open(settings_file, "r") as f:
        settings = json.load(f)

    schema = json.load(resource_stream("baco", "schema.json"))

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    try:
        DefaultValidatingDraft4Validator(schema).validate(settings)
    except Exception as ve:
        print("Failed to validate json:")
        # print(ve)
        raise ve

    settings["log_file"] = add_path(settings, settings["log_file"])

    return settings


def extend_with_default(validator_class):
    """
    Initialize the json schema with the default values declared in the schema.json file.

    Input:
         - validator_class:
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])
        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return validators.extend(validator_class, {"properties": set_defaults})


def validate_json(parameters_file):
    """
    Validate a json file using BaCO's schema.

    Input:
         - paramters_file: json file to validate.
    Returns:
        - dictionary with the contents from the json file
    """
    filename, file_extension = os.path.splitext(parameters_file)
    assert file_extension == ".json"
    with open(parameters_file, "r") as f:
        settings = json.load(f)

    schema = json.load(resource_stream("baco", "schema.json"))

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    DefaultValidatingDraft4Validator(schema).validate(settings)

    return settings


#####################################
# GENERAL
#####################################

def add_path(settings:Dict, file_name:str):
    """
    Add run_directory if file_name is not an absolute path.

    Input:
         - run_directory:
         - file_name:
    Returns:
        - the correct path of file_name.
    """
    if file_name[0] == "/":
        return file_name
    else:
        if settings["run_directory"] == "":
            return str(file_name)
        else:
            return os.path.join(settings["run_directory"], file_name)


def set_output_data_file(settings, headers):
    """
    Set the csv file where results will be written. This method checks
    if the user defined a custom filename. If not, it returns the default.
    Important: if the file exists, it will be overwritten.

    Input:
         - given_filename: the filename given in the settings file.
         - run_directory: the directory where results will be stored.
         - application_name: the name given to the application in the settings file.
    """
    if settings["output_data_file"] == "output_samples.csv":
        settings["output_data_file"] = settings["application_name"] + "_" + settings["output_data_file"]
    settings["output_data_file"] = add_path(
        settings, settings["output_data_file"]
    )
    with open(settings["output_data_file"], "w") as f:
        w = csv.writer(f)
        w.writerow(headers)


#####################################
# SAVE AND LOAD
#####################################

def load_data_file(
    space: Space,
    data_file: str,
    selection_keys_list=[],
    only_valid=False,
) -> DataArray:
    """
    This function read data from a csv file.

    Input:
         - data_file: the csv file where the data to be loaded resides.
         - selection_keys_list: contains the key columns of the csv file to be filtered.
    Returns:
        - data_array
    """
    with open(data_file, "r") as f_csv:
        data = list(csv.reader(f_csv, delimiter=","))
    data = [i for i in data if len(i) > 0]
    number_of_points = len(data) - 1
    headers = data[0]  # The first row contains the headers
    headers = [header.strip() for header in headers]
    data = [row for idx, row in enumerate(data) if idx != 0]
    # Check correctness
    for parameter_name in space.input_output_parameter_names:
        if parameter_name not in headers:
            raise Exception(
                f"Error: when reading the input dataset file the following entry was not found in the dataset but declared as a input/output parameter: {parameter_name}"
            )

    # make sure that the values are in the correct order
    parameter_indices = [headers.index(parameter_name) for parameter_name in space.parameter_names if parameter_name in selection_keys_list or not selection_keys_list]
    parameters_array = space.convert([[row[i] for i in parameter_indices] for row in data], from_type="string", to_type="internal")

    metric_indices = [headers.index(metric_name) for metric_name in space.metric_names]
    metrics_array = torch.tensor([[float(row[i]) for i in metric_indices] for row in data], dtype = torch.float64)
    if space.enable_feasible_predictor:
        feasible_array = torch.tensor([row[headers.index(space.feasible_output_name)] == space.true_value for row in data], dtype = torch.bool)
    else:
        feasible_array = torch.Tensor()
    if "timestamp" in headers:
        timestamp_array = torch.tensor([float(row[headers.index("timestamp")]) for row in data], dtype = torch.float64)
    else:
        timestamp_array = torch.zeros(parameters_array.shape[0], dtype=torch.float64)

    data_array = DataArray(parameters_array, metrics_array, timestamp_array, feasible_array)
    # Filtering the valid rows
    if only_valid:
        data_array = data_array.get_feasible()

    return data_array


def load_data_files(space, filenames, selection_keys_list=[], only_valid=False):
    """
    Create a new data structure that contains the merged info from all the files.

    Input:
         - filenames: the input files that we want to merge.
    Returns:
        - an array with the info in the param files merged.
    """
    arrays = (load_data_file(space, filename, selection_keys_list=selection_keys_list, only_valid=only_valid)[:-1] for filename in filenames)
    data_array = arrays[0]
    for array in arrays[1:]:
        data_array.cat(arrays)
    return data_array

def load_previous(space:Space, settings:Dict) -> Tuple[torch.Tensor, List[str], int, int]:
    """
    Loads a data from a previous to run to be continued.
    """

    if not settings["resume_optimization_file"].endswith(".csv"):
        raise Exception("Error: resume data file must be a CSV")
    if settings["resume_optimization_file"] == "output_samples.csv":
        settings["resume_optimization_file"] = settings["application_name"] + "_output_samples.csv"

    data_array = load_data_file(space, settings["resume_optimization_file"] )
    absolute_configuration_index = data_array.len  # get the number of points evaluated in the previous run
    beginning_of_time = data_array.timestamp_array[-1]  # Set the timestamp back to match the previous run
    print(
        "Resumed optimization, number of samples = %d ......."
        % absolute_configuration_index
    )
    return data_array, absolute_configuration_index, beginning_of_time
