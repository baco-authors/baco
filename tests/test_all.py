import os

import baco  # noqa
from aux.functions import *
from aux.test_cli import branin4_cli

from typing import Callable
from baco.util.file import read_settings_file

testing_directory = os.path.dirname(__file__)

if not os.path.isdir(os.path.join(f"{testing_directory}", "logs")):
    os.mkdir(os.path.join(f"{testing_directory}", "logs"))

def runBenchmark(scenario:str, function:Callable):
    print(scenario)
    settings_file = os.path.join(f"{testing_directory}", "aux", f"{scenario}.json")
    settings = read_settings_file(settings_file)
    settings['log_file'] = os.path.join(f"{testing_directory}","logs", f"{scenario}.log")
    settings["output_data_file"] = os.path.join(
        f"{testing_directory}", "outputfiles", f"{scenario}.csv"
    )
    settings['resume_optimization_file'] = os.path.join(f"{testing_directory}", "aux", f"{settings['resume_optimization_file']}")
    baco.optimize(settings, function)

###################
# GP
###################
print("GP")
runBenchmark("branin4_scenario_gp", branin4_function)

###################
# RF
###################
print("RF")
# requires scikit-learn>=1.3.0
# runBenchmark("branin4_scenario_rf", branin4_function)

###################
# Integer
###################
print("Integer")
runBenchmark("branin4_scenario_integer", branin4_function)

###################
# Resume
###################
print("Resume")
runBenchmark("branin4_scenario_resume", branin4_function)

###################
# Constraints
###################
print("Constraints")
runBenchmark("branin4_scenario_feas", branin4_function_feas)

###################
# DISCRETE
###################
print("Discrete")
runBenchmark("branin4_scenario_discrete", branin4_function)
runBenchmark("rs_cot_1024_scenario", rs_cot_1024)

###################
# BOPRO
###################
print("BOPRO")
#runBenchmark("branin4_scenario_bopro", branin4_function)

###################
# PIBO
###################
print("PIBO")
#runBenchmark("branin4_scenario_pibo", branin4_function)


###################
# PERM
###################
print("Perm")
runBenchmark("perm", perm)

###################
# extra methods
###################

settings_file = os.path.join(f"{testing_directory}", "aux", "rs_cot_1024_scenario.json")
settings = read_settings_file(settings_file)
settings['log_file'] = os.path.join(f"{testing_directory}", "logs", "rs_cot_1024_scenario.log")
settings['output_data_file'] = os.path.join(f"{testing_directory}", "outputfiles", "rs_cot_1024_scenario.csv")

print("OpenTuner")
settings["optimization_method"] = "opentuner"
settings['log_file'] = os.path.join(f"{testing_directory}", "logs", "rs_cot_1024_scenario_OT.log")
settings['output_data_file'] = os.path.join(f"{testing_directory}", "outputfiles", "rs_cot_1024_scenario_OT.csv")
baco.optimize(settings, rs_cot_1024)

print("Ytopt")
settings["optimization_method"] = "ytopt"
settings['log_file'] = os.path.join(f"{testing_directory}", "logs", "rs_cot_1024_scenario_yt.log")
settings['output_data_file'] = os.path.join(f"{testing_directory}", "outputfiles", "rs_cot_1024_scenario_yt.csv")
baco.optimize(settings, rs_cot_1024)

print("Ytopt_CCS")
settings["optimization_method"] = "ytopt_ccs"
settings['log_file'] = os.path.join(f"{testing_directory}", "logs", "rs_cot_1024_scenario_ccs.log")
settings['output_data_file'] = os.path.join(f"{testing_directory}", "outputfiles", "rs_cot_1024_scenario_ccs.csv")
baco.optimize(settings, rs_cot_1024)


###################
# RS
###################
settings_file = os.path.join(f"{testing_directory}", "aux", "branin4_scenario_gp.json")
settings = read_settings_file(settings_file)
settings['log_file'] = os.path.join(f"{testing_directory}", "logs", "branin4_scenario_gp_rs.log")
settings['output_data_file'] = os.path.join(f"{testing_directory}", "outputfiles", "branin4_scenario_gp_rs.csv")
settings["design_of_experiment"]["number_of_samples"] = 30
settings["optimization_iterations"] = 0
print("RS")
baco.optimize(settings, branin4_function)

###################
# CLI
###################
print("cli")
branin4_cli(testing_directory)

