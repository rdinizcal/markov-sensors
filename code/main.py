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
def filterRecords(records, age_range=None, bmi_range=None, icu_type=None):
    """
    Filters records based on age, BMI, and ICU type, then collects vital signals.
    
    Parameters:
    - records: List of Record objects to filter.
    - age_range: Tuple (min_age, max_age) or None to skip age filtering.
    - bmi_range: Tuple (min_bmi, max_bmi) or None to skip BMI filtering.
    - icu_type: String specifying ICU type or None to skip ICU type filtering.
    
    Returns:
    - Dictionary of signals grouped by name: {signal_name: [values]}.
    """
    X = {}

    for record in records:
        # Filter by ICU type
        if icu_type and record.icutype != icu_type:
            continue

        # Filter by age range
        if age_range:
            age = int(record.age)
            if not (age_range[0] <= age <= age_range[1]):
                continue

        # Filter by BMI range
        weight_kg = float(record.weight)
        height_m = float(record.height) / 100
        if weight_kg == 0 or height_m == 0:
            continue
        bmi = weight_kg / (height_m * height_m)
        if bmi_range and not (bmi_range[0] <= bmi <= bmi_range[1]):
            continue

        # Collect vital signals
        for name, signal in record.vital_signals.items():
            if not signal:
                continue
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

def get_value(param, records):
    """Helper function to extract a value for a specific parameter."""
    return records.get(param, None)

def parse_file(filepath):
    """Parse a file and return a Record object if age is valid."""
    with open(filepath, "r") as file:
        lines = file.readlines()
    
    # Remove the header
    lines = lines[1:]
    
    # Parse the file into a dictionary
    records = {}
    for line in lines:
        _, parameter, value = line.strip().split(",")
        records[parameter] = value
    
    # Extract parameters from the dictionary
    recordID = get_value("RecordID", records)
    age = get_value("Age", records)
    gender = get_value("Gender", records)
    height = get_value("Height", records)
    icutype = get_value("ICUType", records)
    weight = get_value("Weight", records)

    # Ensure age is provided and check the condition
    if age is not None and int(age) < 16:
        print(f"Underage record found: {recordID}")
        return None
    
    # Create and return a Record object
    return Record(recordID, age, gender, height, icutype, weight), lines

'''
    Input: Data path
    Output: DTMC
'''
def main():
    records = []  # List of Record objects
    vital_signals = ['HR', 'Temp', 'SaO2', 'NIDiasABP', 'NISysABP']  # Signals to process

    '''
    PRE-PROCESS
    '''
    directory_path = input("Enter the path to the directory containing the records: ").strip()
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                record, lines = parse_file(file_path)
            except Exception as e:
                print(f"Error parsing file {filename}: {e}")
                continue

            # Read and append vital signals to the record
            for line in lines:
                for signal in vital_signals:
                    if re.search(signal, line):
                        key = signal
                        _, _, value = line.strip().split(",")
                        if key in record.vital_signals:
                            record.vital_signals[key].append(value)
                        else:
                            record.vital_signals[key] = [value]

            # Insert record into the list of records
            records.append(record)

    '''
    BUILD DTMCs
    '''
    # Define categories
    AGE_CATEGORIES = {"16-29": (16, 29), "30-59": (30, 59), "60+": (60, 120)}
    BMI_CATEGORIES = {
        "Underweight": (0, 18.4), "Normal weight": (18.5, 24.9),
        "Overweight": (25, 29.9), "Obesity 1": (30, 34.9),
        "Obesity 2": (35, 39.9), "Obesity 3": (40, float('inf'))
    }
    ICU_TYPES = ["Cardiac Surgery Unit", "Coronary Surgery Unit", "Medical ICU", "Surgical ICU"]

    base_output_dir = "../results"

    # Generate DTMCs for each category
    for category_name, filter_criteria in [
        ("by age", AGE_CATEGORIES),
        ("by BMI", BMI_CATEGORIES)
    ]:
        for subcategory, criterion in filter_criteria.items():
            if category_name == "by age":
                filtered_records = filterRecords(records, age_range=criterion)
            elif category_name == "by BMI":
                filtered_records = filterRecords(records, bmi_range=criterion)

            # Create directory for the category if it doesn't exist
            output_dir = os.path.join(base_output_dir, category_name, subcategory)
            os.makedirs(output_dir, exist_ok=True)

            for signal in vital_signals:
                states = hardcoded_states(signal)
                mc = MarkovChain(states)

                Xu = filtered_records
                if signal not in Xu:
                    continue

                prevState = states[0]
                for sample in Xu[signal]:
                    for currState in states:
                        if currState.contains(float(sample)):
                            mc.addTrasition(int(prevState.identifier), int(currState.identifier))
                            prevState = currState
                            break

                mc.normalize()

                # Save DTMC results in the specified format
                output_file_path = os.path.join(output_dir, f"{signal}_mc.txt")
                with open(output_file_path, "w+") as outputFile:
                    # Write states
                    outputFile.write("States: ")
                    for state in mc.states:
                        outputFile.write(f"\t{state.identifier}: [{state.lowerBound},{state.upperBound}]")
                    outputFile.write("\n\n")

                    # Write transitions
                    total_transitions = int(sum(sum(row) for row in mc.transitionMatrix))
                    outputFile.write(f"Transitions: {total_transitions}\n\n")

                    # Write transition matrix
                    outputFile.write("Transition Matrix: \n")
                    for row in mc.normalizedTransitionMatrix:
                        formatted_row = " ".join(f"{value:.2f}" for value in row)
                        outputFile.write(f"[{formatted_row}]\n")

                print(f"DTMC for {signal} saved in {output_file_path}")

    # Special handling for ICU types (list instead of dict)
    for icu_type in ICU_TYPES:
        filtered_records = filterRecords(records, icu_type=icu_type)

        output_dir = os.path.join(base_output_dir, "by ICU", icu_type)
        os.makedirs(output_dir, exist_ok=True)

        for signal in vital_signals:
            states = hardcoded_states(signal)
            mc = MarkovChain(states)

            Xu = filtered_records
            if signal not in Xu:
                continue

            prevState = states[0]
            for sample in Xu[signal]:
                for currState in states:
                    if currState.contains(float(sample)):
                        mc.addTrasition(int(prevState.identifier), int(currState.identifier))
                        prevState = currState
                        break

            mc.normalize()

            # Save DTMC results in the specified format
            output_file_path = os.path.join(output_dir, f"{signal}_mc.txt")
            with open(output_file_path, "w+") as outputFile:
                # Write states
                outputFile.write("States: ")
                for state in mc.states:
                    outputFile.write(f"\t{state.identifier}: [{state.lowerBound},{state.upperBound}]")
                outputFile.write("\n\n")

                # Write transitions
                total_transitions = sum(sum(row) for row in mc.transitionMatrix)
                outputFile.write(f"Transitions: {total_transitions}\n\n")

                # Write transition matrix
                outputFile.write("Transition Matrix: \n")
                for row in mc.normalizedTransitionMatrix:
                    formatted_row = " ".join(f"{value:.2f}" for value in row)
                    outputFile.write(f"[{formatted_row}]\n")

            print(f"DTMC for {signal} saved in {output_file_path}")

    print("\nDTMC generation complete.")

if __name__ == "__main__":
    main()
