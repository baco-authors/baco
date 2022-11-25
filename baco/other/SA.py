##########################################
# NOT USED CURRENTLY
#########################################








import sys
import numpy as np
import json
from jsonschema import exceptions, Draft4Validator
from pkg_resources import resource_stream

from baco.param import space
from baco.util.util import (
    Logger,
    extend_with_default,
    get_min_configurations,
    get_min_feasible_configurations,
)


def is_feasible(param_list, point):
    for param in param_list:
        if not param.evaluate_constraints(
            dict(zip([p.name for p in param_list], point))
        ):
            return False
    return True


def SA(parameters_file, black_box_function):

    with open(parameters_file, "r") as f:
        config = json.load(f)

    schema = json.load(resource_stream("baco", "schema.json"))

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    try:
        DefaultValidatingDraft4Validator(schema).validate(config)
    except exceptions.ValidationError as ve:
        print("Failed to validate json:")
        print(ve)
        raise SystemExit

    param_space = space.Space(config)
    parameters = [
        parameter for parameter in param_space.parameters
    ]
    current_point = [p.get_default() for p in parameters]
    current_obj_val = black_box_function(
        dict(zip(param_space.param, current_point))
    )["runtime"]
    best_obj_val = current_obj_val
    best_point = [x for x in current_point]
    infeasible_counter = 0

    T_start = 30
    T_end = 1
    iterations = 10000
    for i in range(iterations):
        if i % 1000 == 0:
            print(f"iter {i}: best_obj_val = {best_obj_val}")
        cont = True
        while cont:
            cont = False
            changing_parameter_idx = np.random.randint(len(parameters))
            old_val = current_point[changing_parameter_idx]
            parameter_name = param_space.parameter_names[changing_parameter_idx]
            parameter = param_space.paramters[changing_parameter_idx]
            if param_space.parameter_types[changing_parameter_idx] == "real":
                new_val = min(
                    parameter.max_value,
                    max(
                        parameter.min_value,
                        0.1
                        * (2 * np.random.random() - 1)
                        * (parameter.max_value - parameter.min_value)
                        + current_point[changing_parameter_idx],
                    ),
                )
            elif param_space.parameter_types[changing_parameter_idx] in ["integer", "ordinal"]:
                idx = parameter.values.index(current_point[changing_parameter_idx])
                if idx == 0:
                    new_idx = 1
                elif idx == len(parameter.values) - 1:
                    new_idx = idx - 1
                else:
                    new_idx = idx + np.random.choice([-1, 1])
                new_val = parameter.values[new_idx]
            else:
                new_val = np.random.choice(parameter.values)
            current_point[changing_parameter_idx] = new_val
            if not is_feasible(parameters, current_point):
                cont = True
                current_point[changing_parameter_idx] = old_val

        obj_val = black_box_function(
            dict(zip(param_space.parameter_names, current_point))
        )["runtime"]
        if (
            np.exp(
                -(obj_val - current_obj_val)
                / (T_end + T_start * (((iterations - i) / iterations) ** 2))
            )
            > np.random.random()
        ):
            if obj_val < best_obj_val:
                best_obj_val = obj_val
                best_point = [x for x in current_point]

        else:
            current_point[changing_parameter_idx] = old_val

    print(f"best objective val: {best_obj_val}. Best_point: {best_point}.")
    print(f"n_infeasible: {infeasible_counter}")


def OT(parameters_file, black_box_function):

    with open(parameters_file, "r") as f:
        config = json.load(f)

    schema = json.load(resource_stream("baco", "schema.json"))

    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    try:
        DefaultValidatingDraft4Validator(schema).validate(config)
    except exceptions.ValidationError as ve:
        print("Failed to validate json:")
        print(ve)
        raise SystemExit

    param_space = space.Space(config)
    parameters = [
        parameter for parameter in param_space.parameters
    ]
