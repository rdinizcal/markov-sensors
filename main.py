import os
import re
import sys

import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.spatial import distance
from sklearn.cluster import KMeans


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

class Cluster :

    def __init__ (self, label, samples, centroid):
        self.label = label
        self.samples = samples
        self.centroid = centroid
    
    def __str__(self):
        return str(self.label) + ": " + str(self.centroid)

class ClusterStat:

    def __init__ (self, cluster):
        self.cluster = cluster

        self.min = min(cluster.samples)
        self.max = max(cluster.samples)

        if(self.min > self.max) : raise ValueError('Min: ' + str(self.min) + '> Max: ' + str(self.max)) 

        self.size = len(cluster.samples)

    def __str__(self):
        return "Cluster " + str(self.cluster.label) + ": [" + str(self.min) + "," + str(self.max) + "]" + "\nCentroid: " + str(round(self.cluster.centroid,4))

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

    records = [] # records list

    # pre-process data
    if exe[0] is 'p':

        input_path = sys.argv[2]

        files = os.listdir(input_path)
        if not files: 
            raise FileNotFoundError("Empty input folder, try again.")
        
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
    if len(records) < 1 : 
        print("No record found, clustering process was terminated.")
        return  

    # put vital signals of every record in X structure
    X = {str : list}
    for record in records :
        for name,signal in record.vital_signals.items() :
            if name in X:
                if signal : X[name].extend(signal)
            else:
                X[name] = signal

    Xu = X # unsorted X, used for markov chain
    # sort all vital signals
    (X[name].sort() for name in X)

    for signal in vital_signals:
        xArr = np.array(X[signal], dtype=float)
        xArr = xArr.reshape(-1,1)

        within_ss   = {}
        total_ss    = {}
        between_ss  = {}
        stats = {int:list} # cluster numebr : cluster stats list
        
        clustRange = range(1,11) # [1,...,10]
        for k in clustRange :
            '''
            Compute clusters
            '''
            # Compute kmeans with k clusters
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=1000, verbose=0).fit(xArr)

            # Inter- and Intra- clustering sum of squared distances
            # Total within-cluster sum of squares
            within_ss[k]    = kmeans.inertia_
            # Total sum of squares
            total_ss[k]     = sum(distance.sqeuclidean(xArr, cent) for cent in kmeans.cluster_centers_)
            # Total between-cluster sum of squares  
            between_ss[k]   = total_ss[k] - within_ss[k]

            '''
            Statistical analysis
            '''

            centers = kmeans.cluster_centers_
            centers = [item for sublist in centers for item in sublist]

            # Builds dictionary {label1: [sample1, sample2, sample3], label2: [sample1, sample2, sample3]}
            clust_data = {}
            for sample_idx in range(0,len(X[signal])):
                label = kmeans.labels_[sample_idx]
                sample = float(X[signal][sample_idx])
                if label in clust_data:
                    clust_data[label].append(sample)
                else:
                    clust_data[label] = [sample]
            
            clusters = []
            for label in clust_data:
                cluster = Cluster(int(label),clust_data[label],0.0)
                clusters.append(cluster)

            for i in range(0,len(clusters)):
                clusters[i].centroid = centers[i]
            
            # sort clusters
            clusters.sort(key=lambda x: x.centroid)

            # relabeling clusters to crescent order
            for i,cluster in enumerate(clusters): cluster.label = i

            statList = [ClusterStat(cluster) for cluster in clusters]
            
            '''
            Display data
            '''
            print("\nFor n_clusters = ", k)
            for s in statList: print(s)

            stats[k] = statList
        
        '''
        Plot elbow graph for clustering size choice
        '''
        plt.figure()

        # Plot intra-cluster
        #intra_clust = np.array(list(within_ss.values()))/np.array(list(total_ss.values()))
        intra_clust = np.array(list(within_ss.values()))
        intra_clust = [x for x in intra_clust]
        plt.plot(clustRange, intra_clust, 'b.-', label='Intra-cluster') # within cluster

        #plt.ylim((0,100))
        plt.grid(True)
        plt.locator_params(axis="x", nbins=10)
        plt.xlabel('Number of clusters')
        plt.locator_params(axis="y", nbins=10)
        plt.ylabel('Inertia')
        plt.title('Elbow method for KMeans clustering for ' + signal + ' signal')
        plt.legend()

        plt.draw()
        plt.pause(0.001)

        kIdx = int(input("\nOptimal number of clusters: "))
        clustStatsList = stats[kIdx]

        #plt.ylim((0,100))
        plt.grid(True)
        plt.locator_params(axis="x", nbins=10)
        plt.xlabel('Number of clusters')
        plt.locator_params(axis="y", nbins=10)
        plt.ylabel('Inertia')
        plt.title('Elbow method for KMeans clustering for ' + signal + ' signal')
        plt.legend()

        # Mark the chosen cluster number
        plt.plot(clustRange[kIdx-1], intra_clust[kIdx-1], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')

        # Plot spectral
        centroids = np.array([clustStat.cluster.centroid for clustStat in clustStatsList])

        lArr = []
        for sample in xArr[:, 0]:
            for clustStat in clustStatsList:
                if sample >= clustStat.min and sample <= clustStat.max:
                    lArr.append(clustStat.cluster.label) 

        labels = np.array(lArr)

        plt.figure()

        cmap = cm.get_cmap('tab10')
        colors = cmap(labels)
        
        plt.scatter(xArr[:, 0], [0]*len(xArr), marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Draw white circles at cluster centroids
        plt.scatter(centroids, [0]*len(centroids), marker='o', c="white", alpha=1, s=100, edgecolor='k')

        for i, c in enumerate(centroids):
            lab = str(round(float(c),2)) + ": [" + str(clustStatsList[i].min) + "," + str(clustStatsList[i].max) + "]"
            plt.scatter(c, 0, marker='$%d$' % i, alpha=1,s=25, edgecolor='k', label=lab)

        plt.title("Clustered data with " +  str(kIdx) + " clusters.")
        plt.yticks([])
        plt.xlabel("Feature space for heart rate")
        plt.legend()

        # Plot vertical bar (expect normal distribution) 
        plt.figure()
        for i, c in enumerate(centroids):
            plt.bar(str(i), clustStatsList[i].size, width=1.0, align='center', alpha=0.5)
        plt.ylabel('Instances')
        plt.xlabel('Cluster label')
        plt.title('Clustered instances distribution')
        
        plt.draw()
        plt.pause(0.001)

        input("\nPress [enter] to build the markov chains.")

        # build markov chain
        states = [ State() for i in range(len(clustStatsList))]
        for i,clustStat in enumerate(clustStatsList):
            states[i].identifier = clustStat.cluster.label
            states[i].lowerBound = clustStat.min
            states[i].upperBound = clustStat.max

        mc = MarkovChain(len(states))

        prevState = states[0]
        for sample in Xu[signal]:
            for currState in states:
                if currState.contains(float(sample)) : 
                    mc.addTrasition(int(prevState.identifier),int(currState.identifier)) 
                    prevState = currState
                    break

            #if not flag : raise Exception (str(sample) + " couldnt fit into any state!")

        mc.normalize()

        print("\nStates: " + str(mc.states))
        print("\nTransitions: " + str(mc.transitions))
        print("\nTransition Matrix: \n" + str(mc.transitionMatrix))
        print("\nNormalized Transition Matrix: \n" + str(mc.normalizedTransitionMatrix))

        outputFilename = str(signal)+"mc.txt"
        outputFile = open(outputFilename, "w+")
        outputFile.write(str(mc.normalizedTransitionMatrix))
        outputFile.close()

if __name__ == "__main__":
    main()
