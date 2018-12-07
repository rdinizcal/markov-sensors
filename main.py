import os
import re
import sys

import numpy as np

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
        return str(self.label) 

class ClusterStat:

    def __init__ (self, cluster):
        self.cluster = cluster

        self.min = min(cluster.samples)
        self.max = max(cluster.samples)

        if(self.min > self.max) : raise ValueError('Min: ' + str(self.min) + '> Max: ' + str(self.max)) 

        self.size = len(cluster.samples)

    def __str__(self):
        return "Cluster " + str(self.cluster.label) + ": [" + str(self.min) + "," + str(self.max) + "]" + "\nCentroid: " + str(round(self.cluster.centroid,4))

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

    # sort all vital signals
    (X[name].sort() for name in X)

    for signal in vital_signals:
        xArr = np.array(X[signal], dtype=float)
        xArr = xArr.reshape(-1,1)

        within_ss   = {}
        total_ss    = {}
        between_ss  = {}
        
        Kclust = range(1,6) # [1,...,10]
        for k in Kclust :
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
            
            # list of clusters
            clusters = []
            for label in clust_data:
                cluster = Cluster(int(label),clust_data[label],0.0)
                clusters.append(cluster)
            
            # sort list
            centers.sort()
            clusters.sort(key=lambda x: x.label)

            for i in range(0,len(clusters)):
                clusters[i].centroid = centers[i]

            stats = [ClusterStat(cluster) for cluster in clusters]
            
            '''
            Display data
            '''
            print("\nFor n_clusters = ", k)
            for s in stats: print(s)

            # build markov chain


if __name__ == "__main__":
    main()
