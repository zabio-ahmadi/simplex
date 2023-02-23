from lib import *

problems = [
    'flow_1.txt',                   # 0
    'lp_glaces.txt',                # 1
    'lp_test_2.txt',                # 2
    'lp_test_non_admissible.txt',   # 3
    'lp_test.txt',                  # 4
    'network01_lp.txt',             # 5
    'network02_lp.txt',             # 6
    'network03_lp.txt',             # 7
    'network04_lp.txt',             # 8
    'network05_lp.txt'              # 9
]

simplex = simplex(problems[3])
simplex.debug(True)
simplex.printSolution()
