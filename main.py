import os
import sys
import re

# TODO : method extraction
def getVal(index, str):
	arr = str.split(",")
	match = re.match("^[-+]?([0-9]+(\\.[0-9]+)?|\\.[0-9]+)$",arr[index])
	return arr[index][match.start():match.end()]

# TODO : method extraction
def getICUType(_type):
	
	type = int(_type) 
	
	if (type == 1):
		answer = "Coronary Care Unit"
	elif (type == 2):
		answer = "Cardiac Surgery Unit"
	elif (type == 3):
		answer = "Medical ICU"
	elif (type == 4):
		answer = "Surgical ICU"
	else :
		answer = "unknown"
		
	return answer

# TODO : unused
def getTime(index, str):
	arr = str.split(",")
	match = re.match("^(?:(?:([01]?\\d|2[0-3]):)?([0-5]?\\d):)?([0-5]?\\d)$",arr[index])
	return arr[index][match.start():match.end()]

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
        
        pr_data = {int : {str : list}}

        vital_signals = ["HR"] # vital signals to be processed

        for f in files:
            print("Reading " + f + "...")
            i_file = open(input_path+f, "r")
            lines = i_file.readlines()
            del(lines[0]) # Time,Parameter,Value

            # read and build header
            header = ['Age','Gender','Height','ICUType','Weight']
            key = (getVal(2,lines[0])) 
            pr_data[key] = {}
            del(lines[0])
            for h in header:
                if h is 'ICUType' : 
                    pr_data[key][h] = getICUType(getVal(2,lines[0]))
                else : 
                    pr_data[key][h] = getVal(2,lines[0])

                del(lines[0])


            # read and build vital signals
            for line in lines:
                for signal in vital_signals:
                    if re.search(signal, line) :
                        signal_key = signal
                        value = getVal(2,line)
                        if signal_key in pr_data[key]:
                            pr_data[key][signal_key].append(value)
                        else :
                            pr_data[key][signal_key] = [value]

        if len(exe) > 1 and exe[1] is 'p':

            output_fname = str(sys.argv[3])
            o_file = open(output_fname+".csv", "w+")
            f_header = "RecordID,Age,Gender,Height,ICUType,Weight,HR\n"
            o_file.write(f_header)

            for key,d in pr_data.items():
                rid = key
                signal = []
                for k,v in d.items():
                    if k is 'Age': age = v
                    elif k is 'Gender': gender 	= v
                    elif k is 'Height': height 	= v
                    elif k is 'ICUType': icutype = v
                    elif k is 'Weight': weight 	= v
                    else: signal = v
                
                if signal is not list:
                    for el in signal:
                        o_file.write(rid + "," + age + "," + gender + "," + height + "," + icutype + "," + weight + "," + el + "\n")
         
    # clustering
    
    # build markov chain


if __name__ == "__main__":
    main()