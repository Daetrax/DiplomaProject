import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def dbscan(points, frame, mask):

    # #############################################################################
    # Generate sample data
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=len(points), centers=centers, cluster_std=0.4,
    #                             random_state=0)
    # X = StandardScaler().fit_transform(X)

    X = [ (point.pt[0], point.pt[1]) for point in points]
    X = np.asarray(X, dtype=np.float64)
    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    eps = 0.15
    min_samples = 10
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # #############################################################################
    # Plot result
    # import matplotlib.pyplot as plt
    #
    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=14)
    #
    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=6)
    #
    # db_attributes = "eps: " + str(eps) + "\nmin_samples: " + str(min_samples)
    #
    # plt.text(-2, 2.6, db_attributes, fontsize=12)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()

    # #############################################################################
    # Show clusters in image
    import random, cv2
    image = frame.copy()
    keypoint_groups = [0] * (n_clusters_ + 1)

    keypoint_groups_hits = [0] * (n_clusters_ + 1)
    keypoint_groups_maximum_hits = [0] * (n_clusters_ + 1)

    # sort keypoints into clusters
    for point, label in zip(points, db.labels_):
        pos = label
        if pos == -1:
            pos = n_clusters_
        if not keypoint_groups[pos]:
            keypoint_groups[pos] = [point]
            keypoint_groups_maximum_hits[pos] = 1
            continue
        keypoint_groups[pos].append(point)
        keypoint_groups_maximum_hits[pos] += 1

        # ccm = image.copy()
        # cv2.drawKeypoints(ccm, [point], ccm, color=(255, 0, 0))
        # cv2.imshow("One key", ccm)
        # cv2.waitKey(0)
        if np.any(mask[int(point.pt[1]), int(point.pt[0])]) == 1:
            # print("Got one cluster in BP")
            if not keypoint_groups_hits[pos]:
                keypoint_groups_hits[pos] = 1
            else:
                keypoint_groups_hits[pos] += 1
        # print(point)
    # print(keypoint_groups_hits)
    # draw clustered keypoints with random color

    for i, clust in enumerate(keypoint_groups):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if i == len(keypoint_groups) - 1:
            color = (0, 0, 0)
        # print(i, " / ", len(keypoint_groups))
        # print(color)
        if not clust:
            continue
        cv2.drawKeypoints(image, clust, image, color=color)


    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    # mc_mask = mask.copy()
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    # cv2.imshow("Clusters", image)
    cv2.waitKey(0)
    return keypoint_groups_hits, keypoint_groups_maximum_hits