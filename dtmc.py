import numpy as np

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