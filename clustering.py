from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from plotting import maximize_screen


def clustering(scaled_features):
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    maximize_screen()
    plt.show()
    kl = KneeLocator(
        range(1, 11), sse, curve="convex", direction="decreasing"
    )
    print('Best number of clusters: {}'.format(k1.elbow))
