import array
import sys
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():

    # Read states and build states array
    inputFilename = str(sys.argv[1])
    #max_clust = int(sys.argv[2])

    iFile = open(inputFilename,"r+")

    lines = iFile.readlines()
    lines.pop(0)

    X = []
    for line in lines:
    	words = line.split(',')
    	X.append(float(words[7]))

    iFile.close()
    
    X.sort()
    ts = X
    X = np.array(X, dtype=float)
    X = X.reshape(-1,1)

    # clusterize
    within_ss = {}
    total_ss = {}
    between_ss = {}
    KK = range(1, 11)
    for k in KK:
        
        '''
        Compute clusters
        '''
        # Compute kmeans with k clusters
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=1000, verbose=0).fit(X)

        # Silhouette
        if(k!=1):
            silhouette_avg = silhouette_score(X, kmeans.labels_, sample_size=10000)

        # Inter- and Intra- clustering sum of squared distances
        centroids = kmeans.cluster_centers_

        within_ss[k]    = kmeans.inertia_                                           # Total within-cluster sum of squares
        total_ss[k]     = sum(distance.sqeuclidean(X, cent) for cent in centroids)  # Total sum of squares
        between_ss[k]   = total_ss[k] - within_ss[k]                                # The between-cluster sum of squares
        
        '''
        Statistical analysis
        '''
        # Builds dictionary {1: [label1: [value1, value2, value3], 2: [label2: [value1, value2, value3]]}
        clust_data_ = {}
        for i in range(0,len(ts)-1):
            key = kmeans.labels_[i]
            value = ts[i]
            if key in clust_data_:
                clust_data_[key].append(value)
            else:
                clust_data_[key] = [value]

        # Statistical analysis of clusters
        size = [0]*k
        max = [0]*k
        min = [float('Inf')]*k
        for key, arr in clust_data_.items():
            for value in arr:
                size[key] += 1
                if value < min[key] : min[key] = value
                if value > max[key] : max[key] = value

        '''
        Display data
        '''
        # Rounding for visualization
        round_centroids = [round(float(cent),4) for cent in centroids]
        if(k!=1):
            round_silhouette = round(silhouette_avg,4)

        # Print on screen the statistical analysis
        print("\nFor n_clusters = ", k)
        for label in range(k) :
            print("Cluster " + str(label) + ": ")
            print("\t[" + str(min[label]) + "," + str(max[label]) + "] within " + str(size[label]) + " instances.")
            print("\tThe centroid is in " + str(round_centroids[label]))

        if(k!=1): 
                print("The average silhouette_score is : " + str(round_silhouette))

        '''
        # Plot spectral
        plt.figure()
        cmap = cm.get_cmap("Spectral")
        colors = cmap(kmeans.labels_.astype(float) / k)
        plt.scatter(X[:, 0], [0]*len(X), marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Draw white circles at cluster centroids
        plt.scatter(centroids[:, 0], [0]*len(centroids), marker='o', c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centroids):
            plt.scatter(c[0], 0, marker='$%d$' % i, alpha=1,s=50, edgecolor='k')

        plt.title("The visualization of the clustered data with " +  str(k) + " clusters.\n Avg silhouette: " + str(round_silhouette))
        plt.set_yticklabels([])
        plt.xlabel("Feature space for the heart rate")
        '''

    # Plot graph for elbow evaluation
    plt.figure()
    intra_clust = np.array(list(between_ss.values()))/np.array(list(total_ss.values()))
    intra_clust = [x*100 for x in intra_clust]
    plt.plot(KK, intra_clust, 'b.-', label='Inter-cluster') # between cluster
    inter_clust = np.array(list(within_ss.values()))/np.array(list(total_ss.values()))
    inter_clust = [x*100 for x in inter_clust]
    plt.plot(KK, inter_clust, 'r.-', label='Intra-cluster') # within cluster
    plt.plot(KK[2], intra_clust[2], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.ylim((0,100))
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance (%)')
    plt.title('Elbow for KMeans clustering')
    plt.legend()

    plt.show()

    # print states out 


if __name__ == "__main__":
    main()
