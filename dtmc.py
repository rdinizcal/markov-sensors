import numpy as np

class MarkovChain:

	def __init__(self, states):
		self.states = states
		self.n_states = len(states)
		self.transitions = 0
		self.transitionMatrix = np.zeros( (self.n_states, self.n_states) )
		self.normalizedTransitionMatrix = np.zeros( (self.n_states, self.n_states) )

	def addTrasition(self, incomingState, outcomingState):
		self.transitions += 1
		self.transitionMatrix[incomingState][outcomingState] += 1

	def normalize(self):	
		lSum = np.zeros( self.n_states )
		for (line,col),value in np.ndenumerate(self.transitionMatrix):	
			lSum[line] += value

		for (line,col),value in np.ndenumerate(self.transitionMatrix):	
			self.normalizedTransitionMatrix[line][col] = round(value/lSum[line],2)
				
class State:

	def __init__(self):
		self.identifier = 0
		self.lowerBound = 0
		self.upperBound = 0
    
	def contains(self,x):
		return True if(self.lowerBound <= x < self.upperBound) else False
	
	def __str__(self):
		return str(self.identifier) + ": [" + str(self.lowerBound) + "," + str(self.upperBound) + "]"