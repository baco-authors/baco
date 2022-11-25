import os
import sys
if not os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baco import baco  # noqa
from tests.aux.functions import *
from tests.aux.test_cli import branin4_cli
import copy
from typing import Callable
from baco.util.file import read_settings_file

testing_directory = os.path.dirname(__file__)

if not os.path.isdir(os.path.join(f"{testing_directory}", "logs")):
    os.mkdir(os.path.join(f"{testing_directory}", "logs"))

def runBenchmark(scenario:str, function:Callable):
    print(scenario)
    settings_file = os.path.join(f"{testing_directory}", "aux", f"{scenario}.json")
    settings = read_settings_file(settings_file)
    settings['log_file'] = os.path.join(f"{testing_directory}","logs", f"{settings['log_file']}")
    settings['resume_optimization_file'] = os.path.join(f"{testing_directory}", "aux", f"{settings['resume_optimization_file']}")
    baco.optimize(settings, function)

###################
# GP
###################
runBenchmark("branin4_scenario_gp", branin4_function)

###################
# RF
###################
runBenchmark("branin4_scenario_rf", branin4_function)

###################
# Integer
###################
runBenchmark("branin4_scenario_integer", branin4_function)

###################
# Resume
###################
runBenchmark("branin4_scenario_resume", branin4_function)

###################
# Constraints
###################
runBenchmark("branin4_scenario_feas", branin4_function_feas)

###################
# DISCRETE
###################
runBenchmark("rs_cot_1024_scenario", rs_cot_1024)

###################
# BOPRO
###################
runBenchmark("branin4_scenario_bopro", branin4_function)

###################
# PIBO
###################
runBenchmark("branin4_scenario_pibo", branin4_function)

###################
# extra methods
###################

settings_file = os.path.join(f"{testing_directory}", "aux", "rs_cot_1024_scenario.json")
settings = read_settings_file(settings_file)
settings['log_file'] = os.path.join(f"{testing_directory}", "logs", f"{settings['log_file']}")
settings["optimization_method"] = "opentuner"
print("OpenTuner")
baco.optimize(settings, rs_cot_1024)
settings["optimization_method"] = "ytopt"
print("Ytopt")
baco.optimize(settings, rs_cot_1024)

###################
# RS
###################
settings_file = os.path.join(f"{testing_directory}", "aux", "branin4_scenario_gp.json")
settings = read_settings_file(settings_file)
settings['log_file'] = os.path.join(f"{testing_directory}", "logs", f"{settings['log_file']}")
settings["design_of_experiment"]["number_of_samples"] = 30
settings["optimization_iterations"] = 0
print("RS")
baco.optimize(settings, branin4_function)



###################
# CLI
###################
print("cli")
branin4_cli(testing_directory)
