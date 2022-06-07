import numpy as np
np.random.seed(42)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
#Test euclidean distance
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 4])
x = euclidean_distance(x1, x2)
print(x)
class KMeans:
    def __init__(self, K=5, maxIters = 500, plot_steps=False):
        self.K = K
        self.maxIters = maxIters
        self.plot_steps = plot_steps
        self.clusters_list = [[] for _ in range (self.K)]
        self.centroids_list = []

    def predict(self, X):
        self.X = X
        self.samplesNumber, self.featuresNumber = X.shape
        #Initialise centroids
        random_centroids_index = np.random.choice(self.samplesNumber, self.K, replace=False)
        self.centroids_list = [self.X[idx] for idx in random_centroids_index]
        #Optimisation
        for _ in range(self.maxIters):
            #update clusters
            self.clusters_list = self._compute_clusters(self.centroids_list)
            #update centroids
            self.old_centroids_list = self.centroids_list #To check if it converges
            self.centroids_list = _centroids_from_clusters(self.clusters) 
            #check if converged
        #return cluster labels

    def _compute_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for index, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(index)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        return np.argmin(distances)

    def _centroids_from_clusters(self, clusters):
        centroids = np.zeros((self.K, self.featuresNumber))
        for idx, cluster in enumerate(clusters):
            mean_cluster = np.mean((self.X[cluster]), axis=0)
            centroids[idx] = mean_cluster
        return centroids
             


    


