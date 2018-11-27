import array
import math
import sys
from copy import deepcopy

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
    stats = {'centroids': [],
             'labels': [],
             'min': [],
             'max': [],
             'size': []}
    KK = range(1, 11)
    for k in KK:
        
        '''
        Compute clusters
        '''
        # Compute kmeans with k clusters
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=1000, verbose=0).fit(X)

        # Silhouette
        #if(k!=1):
        #    silhouette_avg = silhouette_score(X, kmeans.labels_, sample_size=10000)

        # Inter- and Intra- clustering sum of squared distances
        within_ss[k]    = kmeans.inertia_                                                           # Total within-cluster sum of squares
        total_ss[k]     = sum(distance.sqeuclidean(X, cent) for cent in kmeans.cluster_centers_)    # Total sum of squares
        between_ss[k]   = total_ss[k] - within_ss[k]                                                # The between-cluster sum of squares
        
        '''
        Statistical analysis
        '''
        # Builds dictionary {label1: [value1, value2, value3], label2: [value1, value2, value3]}
        clust_data = {}
        for i in range(0,len(ts)-1):
            key = kmeans.labels_[i]
            value = ts[i]
            if key in clust_data:
                clust_data[key].append(value)
            else:
                clust_data[key] = [value]

        
        # Sort
        for xKey in clust_data.keys():
            for yKey in clust_data.keys():
                if(clust_data[yKey][0]>clust_data[xKey][0]) : 
                    auxLst = deepcopy(clust_data[xKey])
                    clust_data[xKey] = deepcopy(clust_data[yKey])
                    clust_data[yKey] = deepcopy(auxLst)

        centers = kmeans.cluster_centers_
        centers = [item for sublist in centers for item in sublist]
        centers.sort()

        # Statistical analysis of clusters
        stats['centroids'].append(centers)
        stats['labels']   .append(kmeans.labels_)
        stats['min']      .append([float('Inf')]*k) 
        stats['max']      .append([0]*k)
        stats['size']     .append([0]*k)
        for key, lst in clust_data.items():
            for value in lst:
                stats['size'][k-1][key] += 1
                if value < stats['min'][k-1][key] : stats['min'][k-1][key] = value
                if value > stats['max'][k-1][key] : stats['max'][k-1][key] = value

        '''
        Display data
        '''
        # Rounding for visualization
        round_centroids = [round(float(cent),4) for cent in centers]
        #if(k!=1):
        #    round_silhouette = round(silhouette_avg,4)

        # Print on screen the statistical analysis
        print("\nFor n_clusters = ", k)
        for label in range(k) :
            print("Cluster " + str(label) + ": ")
            print("\t[" + str(stats['min'][k-1][label]) + "," + str(stats['max'][k-1][label]) + "] contains " + str(stats['size'][k-1][label]) + " instances.")
            print("\tThe centroid is in " + str(round_centroids[label]))

        #if(k!=1): 
        #        print("The average silhouette_score is : " + str(round_silhouette))

    kIdx = 5

    # Plot graph for elbow evaluation
    plt.figure()

    # Plot intra-cluster
    intra_clust = np.array(list(between_ss.values()))/np.array(list(total_ss.values()))
    intra_clust = [x*100 for x in intra_clust]
    plt.plot(KK, intra_clust, 'b.-', label='Inter-cluster') # between cluster

    # Plot inter-cluster
    inter_clust = np.array(list(within_ss.values()))/np.array(list(total_ss.values()))
    inter_clust = [x*100 for x in inter_clust]
    plt.plot(KK, inter_clust, 'r.-', label='Intra-cluster') # within cluster

    # Mark the chosen cluster number
    #plt.axvline(x=KK[kIdx-1], color="black", linestyle="--")
    plt.plot(KK[kIdx-1], intra_clust[kIdx-1], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='k', markerfacecolor='None')
    plt.plot(KK[kIdx-1], inter_clust[kIdx-1], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='k', markerfacecolor='None')
    
    plt.ylim((0,100))
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.locator_params(axis="x", nbins=10)
    plt.locator_params(axis="y", nbins=10)
    plt.ylabel('Percentage of variance (%)')
    plt.title('Elbow for KMeans clustering')
    plt.legend()

    # Plot spectral
    centroids = np.array(stats['centroids'][kIdx-1])
    labels = np.array(stats['labels'][kIdx-1])

    plt.figure()
    cmap = cm.get_cmap("Spectral")
    colors = cmap(labels.astype(float) / float(kIdx-1))
    plt.scatter(X[:, 0], [0]*len(X), marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Draw white circles at cluster centroids
    plt.scatter(centroids, [0]*len(centroids), marker='o', c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centroids):
        lab = str(round(float(c),2)) + ": [" + str(stats['min'][kIdx-1][i]) + "," + str(stats['max'][kIdx-1][i]) + "]"
        plt.scatter(c, 0, marker='$%d$' % i, alpha=1,s=50, edgecolor='k', label=lab)

    plt.title("The visualization of the clustered data with " +  str(kIdx) + " clusters.")
    plt.yticks([])
    plt.xlabel("Feature space for the heart rate")
    plt.legend()
    
    # Plot vertical bar (expect normal distribution) 
    plt.figure()
    for i, c in enumerate(centroids):
        plt.bar(str(i), stats['size'][kIdx-1][i], width=1.0, align='center', alpha=0.5)
    plt.ylabel('Instances')
    plt.xlabel('Cluster label')
    plt.title('Clustered instances distribution')

    plt.show()

    # print states out 


if __name__ == "__main__":
    main()
