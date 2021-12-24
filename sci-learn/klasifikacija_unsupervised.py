import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def show_clusters(points, cluster_labels):
    first_cluster_points = []
    second_cluster_points = []

    for i in range(len(cluster_labels)):
        cluster_index = cluster_labels[i]
        if cluster_index == 0:
            first_cluster_points.append(points[i])
        elif cluster_index == 1:
            second_cluster_points.append(points[i])

    first_cluster_points_x = [point[0] for point in first_cluster_points]
    first_cluster_points_y = [point[1] for point in first_cluster_points]

    second_cluster_points_x = [point[0] for point in second_cluster_points]
    second_cluster_points_y = [point[1] for point in second_cluster_points]

    plt.scatter(first_cluster_points_x, first_cluster_points_y, c='red')
    plt.scatter(second_cluster_points_x, second_cluster_points_y, c='blue')
    plt.show()


x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# plt.scatter(x[:, 0], x[:, 1], s=100, c='red', marker='o', label='setosa')
# plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(x)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

# show_clusters(x, kmeans.labels_)

new_pts = np.array([[0, 0], [12, 3]])
new_pts_labels = kmeans.predict(new_pts)

print(new_pts_labels)
print(kmeans.inertia_)
