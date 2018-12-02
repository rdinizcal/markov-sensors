## Builds a markov chain transition matrix based on the given states
import numpy as np
import sys

class MarkovChain:

	def __init__(self, states):
		self.states = states
		self.transitions = 0
		self.transitionMatrix = np.zeros( (states, states) )
		self.normalizedTransitionMatrix = np.zeros( (states, states) )

	def addTrasition(self, incomingState, outcomingState):
		self.transitions += 1
		self.transitionMatrix[incomingState][outcomingState] += 1

	def normalize(self):	
		lSum = np.zeros( self.states )
		for (line,col),value in np.ndenumerate(self.transitionMatrix):	
			lSum[line] += value

		for (line,col),value in np.ndenumerate(self.transitionMatrix):	
			self.normalizedTransitionMatrix[line][col] = round(value/lSum[line],2)
				
class State:

	# constructor
	def __init__(self):
		self.identifier = 0
		self.lowerBound = 0
		self.upperBound = 0
    
	# methods
	def contains(self,x):
		return True if(self.lowerBound <= x < self.upperBound) else False
	
def main():

	# Read states and build states array
	statesInFilename = str(sys.argv[1])

	sFile = open(statesInFilename,"r+")
	
	sLines = sFile.readlines()
	sList = [ State() for i in range(len(sLines))]

	#id,lowerBound,upperBound
	i = 0
	for line in sLines:
		words = line.split(',')
		sList[i].identifier = words[0]
		sList[i].lowerBound = int(words[1])
		sList[i].upperBound = int(words[2])
		i+=1

	sFile.close()

	# Read transitions and build markov chain
	transInFilename = str(sys.argv[2])

	tFile = open(transInFilename,"r+")
	
	tLines = tFile.readlines()
	tLines.pop(0)

	mc = MarkovChain(len(sList))
	prevState = sList[0]
	for line in tLines:
		words = line.split(',')
		val = float(words[6])

		flag = False
		for currState in sList:
			if currState.contains(val) : 
				#print(str(prevState.identifier) + " -> " + str(currState.identifier))
				mc.addTrasition(int(prevState.identifier),int(currState.identifier)) 
				prevState = currState
				#print("new prevState: " + str(prevState.identifier) + "\n new currState " + str(currState.identifier))
				flag = True
				break

		#if not flag : raise Exception (str(val) + " couldnt fit into any state!") 
	
	mc.normalize()

	print("States: " + str(mc.states))
	print("Transitions: " + str(mc.transitions))
	print("Transition Matrix: \n" + str(mc.transitionMatrix))
	print("Normalized Transition Matrix: \n" + str(mc.normalizedTransitionMatrix))

	outputFilename = str(sys.argv[3])
	outputFile = open(outputFilename, "w+")
	outputFile.write(str(mc.normalizedTransitionMatrix))
	outputFile.close()

if __name__ == "__main__":
    main()