#Find the optimised bit and word distance of a stabiliser code

import optimiser_fns as optim
import numpy as np

#parameters for the optimisation: Number of parallel cores, view command line output, time limit on a single node exploration.
optimiser_params = {"NumCores": 1, "ViewOutput": True, "TimeLimit": 1e+100}

#which code and which logicals to check
code_params = {"Radius": 1, "WhichLogicals":[0,1]}

#Define and read the input files
StabFilename = 'StabilisersR=%d.dat' % (code_params["Radius"])
LogicalFilename = 'LogicalsR=%d.dat' % (code_params["Radius"])

Stabilisers = np.loadtxt(StabFilename, '\t')
Logicals = np.loadtxt(LogicalFilename, '\t')

#if there is only 1 logical test straight away
if len(Logicals.shape) == 1:
    [MinBitLogical, optim_attributes] = optim.optimise_operator(Stabilisers, Logicals)
    MinWordLogical = MinBitLogical
    BitDistance = sum(MinBitLogical)
    WordDistance = sum(MinWordLogical)
    
else:
    BitDistance = []
    WordDistance = []
    for LogicalNum in code_params["WhichLogicals"]:
        LMatrix = np.delete(Logicals, LogicalNum, 0)
        SLMatrix = np.vstack((Stabilisers, LMatrix))
        
        [MinBitLogical, optim_attributes] = optim.optimise_operator(Stabilisers, Logicals[LogicalNum])
        [MinWordLogical, optim_attributes] = optim.optimise_operator(SLMatrix, Logicals[LogicalNum])
        BitDistance.append(sum(MinBitLogical))
        WordDistance.append(sum(MinWordLogical))

                         

