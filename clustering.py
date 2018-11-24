import array
import sys

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():

    # Read states and build states array
    inputFilename = str(sys.argv[1])
   # n_clusters = int(sys.argv[2])

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
    sse = {}
    for k in range(3, 7):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=1000, verbose=0).fit(X)

        clust_data_ = {}
        for i in range(0,len(ts)-1):
            key = kmeans.labels_[i]
            value = ts[i]
            if key in clust_data_:
                clust_data_[key].append(value)
            else:
                clust_data_[key] = [value]

        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
        
        max = [0]*k
        min = [float('Inf')]*k
        for key, arr in clust_data_.items():
            for value in arr:
                if value < min[key] : min[key] = value
                if value > max[key] : max[key] = value

        print("For n_clusters = ", k)
        for label in range(k):
            print("Cluster " + str(label) + ": [" + str(min[label]) + "," + str(max[label]) + "]")

        silhouette_avg = silhouette_score(X, kmeans.labels_, sample_size=10000)
        print("The average silhouette_score is :", silhouette_avg)

        plt.figure()

        cmap = cm.get_cmap("Spectral")
        colors = cmap(kmeans.labels_.astype(float) / k)
        plt.scatter(X[:, 0], [0]*len(X), marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        plt.scatter(centers[:, 0], [0]*len(centers), marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            plt.scatter(c[0], 0, marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        plt.title("The visualization of the clustered data with " +  str(k) + " clusters.")
        plt.xlabel("Feature space for the feature")

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    
    plt.show()

    # print states out 


if __name__ == "__main__":
    main()
