# Optimiser Functions

import sys
import numpy as np
import random
import gurobipy as gur


#creates a random binary error
def create_error(nQubit, nError, params):
    c = np.zeros_like(np.arange(nQubit))
    
    CSS = params["CSS"]
    ErrorType = params["ErrorModel"]
    
    if CSS == True or ErrorType == 'XZ':
        for i in random.sample(range(nQubit), nError):
            c[i] = 1
    elif ErrorType == 'Z':
        for i in random.sample(range(int(nQubit/2),nQubit), nError):
            c[i]=1
    elif ErrorType == 'X':
        for i in random.sample(range(int(nQubit/2)), nError):
            c[i]=1
    elif ErrorType== 'Depolarising':
        for i in random.sample(range(int(nQubit/2)), nError):
            which_pauli = random.randint(1, 3)
            if which_pauli == 1:
                c[i]=1
            elif which_pauli ==2 :
                c[i]=1
                c[i + int(nQubit/2)] = 1
            elif which_pauli == 3:
                c[i + int(nQubit/2)] = 1
            
    return c


#This function returns a syndrome for a random n qubit error
def get_syndrome(NumErrors, params, stabs):
    
    NumQubits = len(stabs[0])
    
    error = create_error(NumQubits, NumErrors, params)
    
    # for for combined XZ stabilisers need to flip the binary error to get the syndrome properly
    if params["CSS"] == False:
        errorflipped = np.array([error[i] for i in range(int(NumQubits / 2), NumQubits)] + [error[i] for i in range(int(NumQubits / 2))])
        Syndrome = np.fmod(np.dot(stabs, errorflipped), 2)
    else:
        Syndrome = np.fmod(np.dot(stabs, error), 2)
    
    return [Syndrome,error]


#this function returns an optimised correction based on a syndrome
def get_correction(syndrome, stab, logicals, ISF, run_params, optim_params):
    
    #get a pure error from the ISF
    pure_error = np.fmod(np.dot(ISF, syndrome), 2)
    
    #stack the stabiliser and logicals to optimise over
    SLMatrix = np.vstack((stab, logicals))
    
    #run optimisation
    (optim_correction, optim_attributes)  = optimise_operator(SLMatrix, pure_error, code_params=run_params, setup_params=optim_params)
    
    #turn into array
    correction = np.asarray(optim_correction)

    return (correction, optim_attributes)


# This function takes the binary vector 'c' and minimises it over the binary matrix 'stab'
# 
def optimise_operator(stab, c, setup_params = {"NumCores": 1, "ViewOutput": True, "TimeLimit": 1e+100}, code_params = {"CSS": False, "DecodingModel": "XZ"}):
    
    # find the number of stabilisers and qubits
    nStab = len(stab)
    nQubit = len(c)

    #define the model
    m = gur.Model("minwt")
    
    #set parameters
    m.setParam('OutputFlag', setup_params["ViewOutput"])  # show cmd line output
    m.setParam(gur.GRB.Param.Threads, setup_params["NumCores"])  # how many cores to use simultaneously
    m.setParam(gur.GRB.Param.TimeLimit, setup_params["TimeLimit"])  # time limit for single iterations in seconds
 
    #define the variables
    x = m.addVars(nStab, vtype=gur.GRB.BINARY, name="x")
    y = m.addVars(nQubit, vtype=gur.GRB.INTEGER, name="y")
    z = m.addVars(nQubit, vtype=gur.GRB.BINARY, name="z")
    t = m.addVars(nQubit, vtype=gur.GRB.INTEGER, name="t")
    
    
    #m.addConstrs(z[i] == c[i] + quicksum(x[j] for j in stab[i])+2*t[i] for i in range(nQubit))
    m.addConstrs(y[i] == c[i] + gur.quicksum(stab[j][i] * x[j] for j in range(nStab)) for i in range(nQubit))
    m.addConstrs(y[i] == 2 * t[i] + z[i] for i in range(nQubit))
    
    # Force correction to be either all X or Z
    if code_params["CSS"] == False:    
        if code_params["DecodingModel"] == 'X':
            m.addConstr(sum(z[i] for i in range(int(nQubit / 2), nQubit)) == 0)
        if code_params["DecodingModel"] == 'Z':
            m.addConstr(sum(z[i] for i in range(int(nQubit / 2))) == 0)
    
    
    #Set the objective, if depolarising error model this is a quadratic obj, linear otherwise
    if code_params["CSS"] == False and code_params["DecodingModel"] == "Depolarising":
        m.setObjective(sum(z[i] + z[i+int(nQubit / 2)] - z[i] * z[i+int(nQubit / 2)]  for i in range(int(nQubit / 2))), gur.GRB.MINIMIZE)
    else:
        m.setObjective(sum(z[i] for i in range(nQubit)), gur.GRB.MINIMIZE)
    
    #update model and run    
    m.update()
    m.optimize()
    
    #read out some attributes from the model incase you want them.
    optim_attributes = {"ObjectiveValue": m.objval, "BestBound": m.objboundc, "NumIterations": m.itercount, "RunTime": m.runtime}
    
    return [[int((z[i].x) % 2) for i in range(nQubit)], optim_attributes]


#find if the correction was a failure by seeing if the net error commutes with the test logicals
def failure_check(error, correction, stabs, logicals, params):
    
    NumQubits = len(stabs[0])
    NumStabs = len(stabs)
    
    #sum error and correction for net error
    net_error = np.fmod(correction + error, 2)
    
    #quick check that the net error gives a trivial syndrome
    if params["CSS"] == False:
        errorflipped = np.array([net_error[i] for i in range(int(NumQubits / 2), NumQubits)] + [net_error[i] for i in range(int(NumQubits / 2))])
        Net_Syndrome = np.fmod(np.dot(stabs, errorflipped), 2)
    else:
        Net_Syndrome = np.fmod(np.dot(stabs, net_error), 2)
    
    if not np.array_equal(Net_Syndrome, np.zeros(NumStabs, dtype=int)):
        sys.exit("The net error is not trivial")
    
    
    #see if the net error commutes with the test logicals.
    if params["CSS"] == False:
        net_error = np.array([net_error[i] for i in range(int(NumQubits / 2), NumQubits)] + [net_error[i] for i in range(int(NumQubits / 2))])
            
    if len(logicals.shape) == 1:
        success = np.fmod(np.dot(net_error, logicals), 2)    
    else:
        success = [np.fmod(np.dot(net_error, logicals[i]), 2) for i in params["TestLogicals"]]    
    
    return success
