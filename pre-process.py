import re
import os
import sys
from enum import Enum

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
	
def getNumValue(index, str):
	arr = str.split(",")
	match = re.match("^[-+]?([0-9]+(\.[0-9]+)?|\.[0-9]+)$",arr[index])
	return arr[index][match.start():match.end()]
	
def getTime(index, str):
	arr = str.split(",")
	match = re.match("^(?:(?:([01]?\d|2[0-3]):)?([0-5]?\d):)?([0-5]?\d)$",arr[index])
	return arr[index][match.start():match.end()]
	
def main():

	outputFilename = str(sys.argv[1])
	prefix = (sys.argv[2])

	outputFile = open(outputFilename+".csv", "w+")
	header = "RecordID,Age,Gender,Height,ICUType,Weight,Time,Heart Rate\n"
	outputFile.write(header)
	
	#filenames = {"142671.csv"}
	filenames = os.listdir(prefix)

	if not filenames: raise FileNotFoundError("No files found in folder (" + prefix + ")")
	
	for filename in filenames:
		
		filename = prefix + filename

		if (re.search(".py",filename) or re.search("output.csv",filename)) : continue
		
		inputFile = open(filename, "r")
		print("Reading file " + filename + "...\n")
		
		lines = inputFile.readlines()
		
		for index in range(7):	
			if(re.search("RecordID", 	lines[index])): 	rid 	= getNumValue(2,lines[index])
			elif(re.search("Age", 		lines[index])): 	age 	= getNumValue(2,lines[index])
			elif(re.search("Gender", 	lines[index])): 	gender 	= getNumValue(2,lines[index])
			elif(re.search("Height", 	lines[index])): 	height 	= getNumValue(2,lines[index])
			elif(re.search("ICUType", 	lines[index])): 	icutype = getNumValue(2,lines[index])
			elif(re.search("Weight", 	lines[index])): 	weight 	= getNumValue(2,lines[index])
		
		for line in lines:
			if(re.search("HR", line)):
				time = getTime(0,line)
				hr = getNumValue(2,line)
				outputFile.write(rid + "," + age + "," + gender + "," + height + "," + getICUType(icutype) + "," + weight + "," + time + "," + hr + "\n")

	
	inputFile.close()
	
	outputFile.close()

if __name__ == "__main__":
    main()