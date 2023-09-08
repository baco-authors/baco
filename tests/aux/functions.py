
import math
import numpy as np

def branin_function(x1, x2):
    """
    Compute the branin function given two parameters.
    The value is computed as defined in https://www.sfu.ca/~ssurjano/branin.html
    :param x1: the first input of branin.
    :param x2: the second input of branin.
    :return: the value of the branin function.
    """
    a = 1.0
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    y_value = a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s

    return y_value


def branin4_function(X):
    """
    Compute the four-dimensional branin function.
    Value in four-dimensional branin is defined as the product of two Brannin functions.
    :param X: dictionary containing the input points.
    :return: the value of the branin function.
    """
    x1 = X["x1"]
    x2 = X["x2"]
    x3 = X["x3"]
    x4 = X["x4"]
    f1_value = branin_function(x1, x2)
    f2_value = branin_function(x3, x4)
    y_value = f1_value * f2_value

    return y_value


def branin4_function_feas(X):
    """
    Compute the four-dimensional branin function.
    Value in four-dimensional branin is defined as the product of two Brannin functions.
    :param X: dictionary containing the input points.
    :return: the value of the branin function.
    """
    x1 = X["x1"]
    x2 = X["x2"]
    x3 = X["x3"]
    x4 = X["x4"]
    f1_value = branin_function(x1, x2)
    f2_value = branin_function(x3, x4)
    y_value = f1_value * f2_value

    return {"Value": y_value, "Valid": (True if x1 > 0 else False)}


def chakong_haimes(X):
    """
    Compute the Chakong and Haimes two-objective function to demonstrate a two-objective optimization example.
    The value is computed as defined in https://en.wikipedia.org/wiki/Test_functions_for_optimization
    :param X: dictionary containing the input points.
    :return: the two values of the Chakong and Haimes function and the feasibility indicator.
    """
    x1 = X["x1"]
    x2 = X["x2"]
    f1_value = 2 + (x1 - 2) * (x1 - 2) + (x2 - 1) * (x2 - 1)
    f2_value = 9 * x1 - (x2 - 1) * (x2 - 1)

    # check constraints
    g1 = x1 * x1 + x2 * x2 <= 225
    g2 = x1 - 3 * x2 + 10 <= 0
    valid = g1 and g2

    output_metrics = {}
    output_metrics["f1_value"] = f1_value
    output_metrics["f2_value"] = f2_value
    output_metrics["Valid"] = valid

    return output_metrics

def rs_cot_1024(X):

    if isinstance(X, dict):
        v3 = X["tuned_v3"]
        v4 = X["tuned_v4"]
        v5 = X["tuned_v5"]
        v6 = X["tuned_v6"]
        v7 = X["tuned_v7"]
        v8 = X["tuned_v8"]

        ls0 = X["tuned_ls0"]
        ls1 = X["tuned_ls1"]
        gs0 = X["tuned_gs0"]
        gs1 = X["tuned_gs1"]

    else:
        v3, v4, v5, v6, v7, v8, ls0, ls1, gs0, gs1 = tuple(X)

    alpha = 0.00001
    beta = 800
    gamma = 10
    delta = 100

    # dummy computation for function value
    function_value = (
        alpha * v3 * v4  # 0 - 1e2
        + alpha * (v5 + v6 - beta) ** 2  # 0 - 1e2
        - gamma * np.sin(v7 / delta) * np.sin(v8 / delta)  # -1e1 - 1e1
        + np.log(ls0 + 1)
        - np.log(ls1 + 1)  # 0 - 1e1
        + np.log(gs0 + 1)
        - np.log(gs1 + 1)  # -1e1 - 0
        + alpha * (v3 + v5 + v7 - ls0 - gs1) ** 2  # 0 - 1e2
    )

    valid = True
    if v3 <= 8:
        valid = False
        function_value = 1e9

    # return {"runtime": function_value}
    return {"runtime": function_value, "Valid": valid}


def perm(X):

    x1 = X["cs"]
    x2 = X["ocs"]
    x3 = X["ont"]
    x4 = X["ost"]
    p = X["p"]
    x5 = X["uf"]

    return {
        "compute_time": x1**2
        + x2**2
        + x3**2
        + x4**2
        + x5**2
        + p[0] ** 2
        + p[1]
        - p[3]
        - p[4] ** 2
        + 2 ** np.random.normal(),
        "Valid": True,
    }
