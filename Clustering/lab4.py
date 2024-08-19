import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
# had to install yellowbrick. initially i thought i was using the wrong kernel but figured it out.

stdDeviationRange = [1, 5, 10, 20, 100]
# differing standard deviation values. i wasnt sure how many i was expected to try.
# i believe 5 values was a safe bet to see the gradual impact of standard deviation.


def generateDataset(stdDeviation):
    x, _ = make_blobs(n_samples=300, n_features=2,
                      cluster_std=stdDeviation, random_state=17)
    # 300 samples with 2 features as defined in the question.
    # standard deviation has multiple values to be tried with.
    # random state was discussed in the first lab if im not wrong, which had to do with the "random" data
    # being reproducable over multiple attempts

    # ensured the data is non-negative and converted it to integers for easier/cleaner represnetation
    x = np.round(x).astype(int)
    x = np.abs(x).astype(int)

    return x


datasets = [generateDataset(std) for std in stdDeviationRange]


def kMeansAlgorithm(k, data):
    # implemented the k-means to customize as per the rubrics
    maxIterations = 50

    # implemented cluster centering randomly as per the rubrics
    centers = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(maxIterations):
        # calculated distances between data points and cluster centers for the next step.
        # this will make it easier to assign the correct centers to each point

        distances = np.linalg.norm(data[:, None] - centers, axis=2)
        points = np.argmin(distances, axis=1)

        for i in range(len(points)):
            # breaking ties in an arbitrary way. eqvidistant points are randomly allocated clusters.
            # im guessing there is more than one way to go about it but the simplest solution made most sense to me.

            minDistanceCenters = np.where(
                distances[i] == distances[i].min())[0]
            if len(minDistanceCenters) > 1:
                points[i] = np.random.choice(minDistanceCenters)

        # updating cluster centers based on the mean of data points assigned to each cluster
        newCenters = np.array([data[points == i].mean(axis=0)
                               for i in range(k)])

        if np.all(centers == newCenters):
            break

        centers = newCenters

    return points, centers


for index, dataset in enumerate(datasets, start=1):
    fig, axs = plt.subplots(2, figsize=(7, 7))

    k = 3
    # i created 3 clusters since the clustering algorithm suggests that 3 clusters is optimal for the range 3-6.
    # if necessary, we can try for more clusters by changing value of k. a loop can also be created.

    labels, _ = kMeansAlgorithm(k, dataset)
    axs[0].scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='viridis')

    # used the silhouette method to determine the optimal number of clusters k.
    # a little unsure why it always suggested 3 (highest silhouetee score)
    model = KMeans(random_state=17)
    visualizer = KElbowVisualizer(
        model, k=(3, 7), metric='silhouette', timings=False)
    visualizer.fit(dataset)
    visualizer.show(ax=axs[1])

    plt.tight_layout()
    plt.close()
