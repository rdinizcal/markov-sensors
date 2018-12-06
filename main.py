import os
import re
import sys


# TODO : method extraction
def getVal(index, str):
	arr = str.split(",")
	match = re.match("^[-+]?([0-9]+(\\.[0-9]+)?|\\.[0-9]+)$",arr[index])
	return arr[index][match.start():match.end()]

# TODO : deprecated
def getTime(index, str):
	arr = str.split(",")
	match = re.match("^(?:(?:([01]?\\d|2[0-3]):)?([0-5]?\\d):)?([0-5]?\\d)$",arr[index])
	return arr[index][match.start():match.end()]


class Record : 

    def __init__ (self, recordID, age, gender, height, icutype, weight):
        self.recordID = recordID
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight

        icutype = int(icutype)
        if (icutype == 1):
            self.icutype = "Coronary Care Unit"
        elif (icutype == 2):
            self.icutype = "Cardiac Surgery Unit"
        elif (icutype == 3):
            self.icutype = "Medical ICU"
        elif (icutype == 4):
            self.icutype = "Surgical ICU"
        else :
            self.icutype = "unknown"
        
        self.vital_signals = {str: []}

'''
    Input: Data path
    Output: DTMC
'''
def main():

    if(len(sys.argv) > 1): 
        exe = sys.argv[1]
    else : 
        print ("Please insert the execution directives:")
        print ("[p input] for pre-processing the input file")
        print ("[pp input out] for printing the pre-processing output at out.csv")
        print ()
        return

    # pre-process data
    if exe[0] is 'p':

        input_path = sys.argv[2]

        files = os.listdir(input_path)
        if not files: 
            raise FileNotFoundError("Empty input folder, try again.")
        
        # initialize pre-processed data
        # exmpl = {RecordID : 
        #           {'Age'         : [24],
        #            'Gender'      : [1],
        #            'Height'      : [173],
        #            'ICUType'     : [3],
        #            'Weight'      : [71.4],
        #            'HR'          : [132, 67, 81, 89, ...]} 

        records = [] # records list

        vital_signals = ['HR'] # vital signals to be processed

        for f in files:
            print("Reading " + f + "...")
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

        if len(exe) > 1 and exe[1] is 'p':
            
            if len(sys.argv) < 4 : 
                raise FileNotFoundError("Output filename not specified")

            output_fname = str(sys.argv[3])
            o_file = open(output_fname+".csv", "w+")
            f_header = "RecordID,Age,Gender,Height,ICUType,Weight,HR\n" 
            o_file.write(f_header)

            for record in records:
                for name,signal in record.vital_signals.items():
                    for el in signal:
                        o_file.write(record.recordID + "," + record.age + "," + record.gender + "," + record.height + "," + record.icutype + "," + record.weight + "," + el + "\n")
    # clustering
    
    # build markov chain


if __name__ == "__main__":
    main()
