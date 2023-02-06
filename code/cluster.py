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
	
