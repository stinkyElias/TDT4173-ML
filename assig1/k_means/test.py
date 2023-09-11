import numpy as np 
import pandas as pd 
import random 

class KMeans:
    
    def __init__(self, k=2):
        self.k = k
        self.centroids = []
        self.retained_centroids = []
        pass
        
    def fit(self, X):
        # devide by 10 because then the scale would be the same for data1 as for data
        if self.k == 10:
            X.loc[:, "x0"] = X.loc[:, "x0"]/10

        previous_distorsion = 100

        #experimented with iterations here. found that 3 is too little and 4 does the trick
        for _ in range(4):
            self.centroids = spawn_cent(X, self.k)
            self.centroids = np.array(self.centroids)

            data_points = X.copy()

            # this array contains the distance between current and the old centroid. 
            difference_in_distance = np.ones(self.k)

            while all(dist>0.0001 for dist in difference_in_distance):

                # copy the value of the old centroid
                previous_centroids = self.centroids.copy()

                # Loop throug every datapoint and connect it to a centroid
                point_clusters=[]
                for number in range(len(data_points)):
                    point_in_data = X[number:number+1]
                    point_in_data = np.array(point_in_data)
                    point_in_data = point_in_data.reshape(-1)
                    distances = []

                    # calculate the distance between a point and all centroids
                    for k in range(self.k):
                        distances.append(euclidean_distance(point_in_data,self.centroids[k]))
                    index_of_min_distance = distances.index(min(distances)) # find the smallest distance, its index is the index of the assigned centroid
                    point_clusters.append(index_of_min_distance) # appends the index of the smallest distance to the point cluster

                data_points["centroids"] = point_clusters

                # take the average position of a cluster to recalculate the centroid position
                for k in range(self.k):
                    self.centroids[k,:] = np.array(data_points[data_points["centroids"]==k][["x0", "x1"]].mean(axis=0))
                    # keep track of the old and new distances by finding the difference
                    difference_in_distance[k] = euclidean_distance(self.centroids[k], previous_centroids[k])

            cluster_assignment = np.array(point_clusters)
            distortion = euclidean_distortion(X, cluster_assignment)
            if distortion < previous_distorsion:
                previous_distorsion = distortion
                self.retained_centroids = self.centroids

    def smallest_distance(self, distances, point_in_data, clusters_to_return):
        for k in range(self.k):
            distances.append(euclidean_distance(point_in_data, self.retained_centroids[k]))
        index_of_smallest_distance = distances.index(min(distances))
        clusters_to_return.append(index_of_smallest_distance)

    def predict(self, X):
        clusters_to_return = []
        # Loop through every point in the data and reshape
        for num in range(len(X)):
            point_in_data = X[num:num+1]
            point_in_data = np.array(point_in_data)
            point_in_data = point_in_data.reshape(-1)

            distances = []
            # Find the distance between each point in the data and all the clustersters. Add the smallest to the list. the index is correct for the centroid
            distances = self.smallest_distance(distances, point_in_data, clusters_to_return)
        return np.array(clusters_to_return)


    def get_centroids(self):
        return self.retained_centroids
            
# --- Some utility functions 


def euclidean_distortion(X, z):

    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):

    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):

    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))

def find_distance_to_closest(X, init_cent):
    dist_to_closest_centroid = []
    for num in range(len(X)):
        distance_to_all_centroids = []
        p = X[num:num+1]
        p = np.array(p)
        p = p.reshape(-1)
        for centroid_number in range(len(init_cent)):
            distance_to_all_centroids.append(euclidean_distance(p, init_cent.iloc[centroid_number,:]))
        dist_to_closest_centroid.append(min(distance_to_all_centroids))
    return dist_to_closest_centroid


def spawn_cent(X, k):
    """
    helper functinon to create centroids
    """
    initial_centroids = (X.sample()).reset_index(drop=True)
    for _ in range(k-1):
        dist_to_closest_centroid = find_distance_to_closest(X, initial_centroids)
        mod_dist_to_closest_centroid = [i**4 for i in dist_to_closest_centroid]
        mod_dist_to_closest_centroid = mod_dist_to_closest_centroid / sum(mod_dist_to_closest_centroid)

        new_centroid = X.sample(1, weights=mod_dist_to_closest_centroid)
        initial_centroids = pd.concat([initial_centroids, new_centroid], ignore_index=True)
    
    return initial_centroids