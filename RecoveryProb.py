import optimiser_fns as optim
import numpy as np


#parameters for the optimisation: Number of parallel cores, view command line output, time limit on a single node exploration.
optimiser_params = {"NumCores": 1, "ViewOutput": True, "TimeLimit": 1e+100}

#General parameters for the simulation: 
    #Radius is just for the input files
    #Set the CSS true if there is only x or z stabilisers
    #The error models are either treat X and Z equally ("XZ"), only X or Z errors ("X","Z") or "Depolarising".
    #Test logicals states which row the logicals you want to check are in
run_params = {"Radius": 1, "CSS": True, "ErrorModel": "XZ", "DecodingModel": "XZ", "TestLogicals": [0], "NumSims": 100}

#Define and read the input files
StabFilename = 'StabilisersR=%d.dat' % (run_params["Radius"])
LogicalFilename = 'LogicalsR=%d.dat' % (run_params["Radius"])
ISFFilename = 'ISFR=%d.dat' % (run_params["Radius"])

Stabilisers = np.loadtxt(StabFilename, '\t')
Logicals = np.loadtxt(LogicalFilename, '\t')
ISF = np.loadtxt(ISFFilename, '\t')

ResultsFilename = 'ResultsR=%d.dat' % (run_params["Radius"])

#iterate over the numer of errors
for NumErrors in range(7):
    failure_list = []
    #iterate over the number of monte carlo samples
    for counter in range(run_params["NumSims"]):
    
        [Syndrome, error] = optim.get_syndrome(NumErrors, run_params, Stabilisers)
        
        [Correction, optim_attributes] = optim.get_correction(Syndrome, Stabilisers, Logicals, ISF, run_params, optimiser_params)
        
        #returns failure binary vector, 0 -> success, 1-> failure
        failure = optim.failure_check(error, Correction, Stabilisers, Logicals, run_params)
        failure_list.append(failure)
        
    #average the failure rate for each test logical    
    failure_mean = np.mean(failure_list, axis = 0)
    
    #append results to file
    with open(ResultsFilename, 'a') as f:
        f.write('%d,' % NumErrors)
        if len(run_params["TestLogicals"]):
            f.write('%f,' % failure_mean)
        else:
            for i in run_params["TestLogicals"]:
                f.write('%f,' % failure_mean[i])
        f.write('%d' % run_params["NumSims"])
        f.write('\n')

