get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
              (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
              (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))

def initialize_centroids(points, k):
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def closest_centroid(points, centroids):
    distances = ((points - centroids[:, np.newaxis])**2).sum(axis = 2)
    return np.argmin(distances, axis = 0)

def move_centroids(points, closest, centroids):
    return np.array([points[closest == k].mean(axis = 0) for k in range (centroids.shape[0])])

def main(points):
    num_iterations = 100
    k = 3
    centroids = initialize_centroids(points, k)
    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)
    
    return centroids


centroids = main(points)

centroids = initialize_centroids(points, 3)

plt.scatter(points[:, 0], points[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
ax = plt.gca()

closest = closest_centroid(points, centroids)
centroids = move_centroids(points, closest, centroids)

plt.scatter(points[:, 0], points[:, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
ax = plt.gca()

