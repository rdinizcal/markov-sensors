import os
import re
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

from cluster import *
from dtmc import *
from record import *


# TODO : method extraction
def getVal(index, str):
	arr = str.split(",")
	match = re.match("^[-+]?([0-9]+(\\.[0-9]+)?|\\.[0-9]+)$",arr[index])
	return arr[index][match.start():match.end()]

'''
    Description: Filter signals to be processed
    Input: records "list(Record)"
    Output: {str:list} "{vital signal name: list of values}"
'''
def selectSignal(records):
    X = {str : list}

    for record in records :
        
        for name,signal in record.vital_signals.items() :
            
            if not signal: continue
            
            for el in signal:
                # Removing outliers
                if float(el) < 0: continue
                if name == "Temp" and float(el) < 24: continue 

                if name in X:
                    X[name].append(el)
                else:
                    X[name] = [el]
    
    return X

'''
    Description: Filter signals to be processed
    Input: records "list(Record)"
    Output: {str:list} "{vital signal name: list of values}"
'''
def filterRecords(records):
    X = {str : list}

    for record in records :

        # Insert here your attribute filtering rule
        #if record.icutype != "Surgical ICU": continue
        #if int(record.age) < 60: continue
        weight_kg = float(record.weight)
        height_m = float(record.height)/100
        if weight_kg == 0 or height_m == 0: continue
        BMI = weight_kg/(height_m*height_m)
        if BMI < 40: continue

        for name,signal in record.vital_signals.items() :
            
            if not signal: continue
            
            for el in signal:

                if name in X:
                    X[name].append(el)
                else:
                    X[name] = [el]
    
    return X

def hardcoded_states(signal):

    if signal == 'HR':
        states = [ State() for i in range(5) ]

        states[0].identifier = 0
        states[0].lowerBound = 0
        states[0].upperBound = 70

        states[1].identifier = 1
        states[1].lowerBound = 70
        states[1].upperBound = 85

        states[2].identifier = 2
        states[2].lowerBound = 85
        states[2].upperBound = 97

        states[3].identifier = 3
        states[3].lowerBound = 97
        states[3].upperBound = 115

        states[4].identifier = 4
        states[4].lowerBound = 115
        states[4].upperBound = 300
    elif signal == "Temp": 
        states = [ State() for i in range(5) ]

        states[0].identifier = 0
        states[0].lowerBound = 0
        states[0].upperBound = 32

        states[1].identifier = 1
        states[1].lowerBound = 32
        states[1].upperBound = 36

        states[2].identifier = 2
        states[2].lowerBound = 36
        states[2].upperBound = 38

        states[3].identifier = 3
        states[3].lowerBound = 38
        states[3].upperBound = 41

        states[4].identifier = 4
        states[4].lowerBound = 41
        states[4].upperBound = 100
    elif signal == 'SaO2':
        states = [ State() for i in range(3) ]

        states[0].identifier = 0
        states[0].lowerBound = 0
        states[0].upperBound = 55

        states[1].identifier = 1
        states[1].lowerBound = 55
        states[1].upperBound = 65

        states[2].identifier = 2
        states[2].lowerBound = 65
        states[2].upperBound = 100
    elif signal == 'NIDiasABP':
        states = [ State() for i in range(3) ]

        states[0].identifier = 0
        states[0].lowerBound = 0
        states[0].upperBound = 80

        states[1].identifier = 1
        states[1].lowerBound = 80
        states[1].upperBound = 90

        states[2].identifier = 2
        states[2].lowerBound = 90
        states[2].upperBound = 300
    elif signal == 'NISysABP':
        states = [ State() for i in range(3) ]

        states[0].identifier = 0
        states[0].lowerBound = 0
        states[0].upperBound = 120

        states[1].identifier = 1
        states[1].lowerBound = 120
        states[1].upperBound = 140

        states[2].identifier = 2
        states[2].lowerBound = 140
        states[2].upperBound = 300

    return states

'''
    Input: Data path
    Output: DTMC
'''
def main():

    records = [] # records list
    vital_signals = ['HR','Temp','SaO2','NIDiasABP','NISysABP'] # vital signals to be processed

    '''
    PRE-PROCESS
    '''
    input_paths = ["data/set-a/", "data/set-b/"]
    count = 1
    for input_path in input_paths:
        
        files = os.listdir(input_path)
        if not files: 
            raise FileNotFoundError("Empty input folder, try again.")

        for f in files:
            print(str(count) + ". reading " + f + "...")
            i_file = open(input_path+f, "r")
            lines = i_file.readlines()
            del(lines[0]) # delete header (Time,Parameter,Value)

            # read and build record object
            recordID = (getVal(2,lines[0])) 
            del(lines[0])
            age = (getVal(2,lines[0])) 
            del(lines[0])
            gender = (getVal(2,lines[0])) 
            del(lines[0]) 
            height = (getVal(2,lines[0])) 
            del(lines[0]) 
            icutype = (getVal(2,lines[0])) 
            del(lines[0]) 
            weight = (getVal(2,lines[0])) 
            del(lines[0]) 

            record = Record(recordID, age, gender, height, icutype, weight)

            # read and append vital signals to record
            for line in lines:
                for signal in vital_signals:
                    if re.search(signal,line) :
                        key = signal
                        value = getVal(2,line)
                        if key in record.vital_signals:
                            record.vital_signals[key].append(value)
                        else :
                            record.vital_signals[key] = [value]

            # insert record into list of records
            records.append(record)
            count+=1

    '''
    BUILD DTMCs
    '''

    for signal in vital_signals:

        states = hardcoded_states(signal)
        mc = MarkovChain(states)
        
        Xu = filterRecords(records)
        if signal not in Xu: continue

        prevState = states[0]
        for sample in Xu[signal]:

            for currState in states:
                if currState.contains(float(sample)) : 
                    mc.addTrasition(int(prevState.identifier),int(currState.identifier)) 
                    prevState = currState
                    break

            #if not flag : raise Exception (str(sample) + " couldnt fit into any state!")

        mc.normalize()

        print("\nStates: ")
        for state in mc.states: print("\t"+str(state))
        print("\nTransitions: " + str(mc.transitions))
        print("\nTransition Matrix: \n" + str(mc.transitionMatrix))
        print("\nNormalized Transition Matrix: \n" + str(mc.normalizedTransitionMatrix))

        outputFilename = str(signal)+"_mc.txt"
        outputFile = open(outputFilename, "w+")
        outputFile.write("States: ")
        for state in mc.states: outputFile.write("\t"+str(state))
        outputFile.write("\n\nTransitions: " + str(mc.transitions))
        outputFile.write("\n\nTransition Matrix: \n" + str(mc.normalizedTransitionMatrix))
        outputFile.close()

        input("\nPress [enter] to continue.")

if __name__ == "__main__":
    main()
