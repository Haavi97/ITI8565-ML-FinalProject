from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from kneed import KneeLocator
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
    silhouette_coefficients = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
        if k>1:
            score = silhouette_score(scaled_features, kmeans.labels_)
            silhouette_coefficients.append(score)
    
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    maximize_screen()
    plt.show()
    
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette")
    maximize_screen()
    plt.show()
    kl = KneeLocator(
        range(1, 11), sse, curve="convex", direction="decreasing"
    )
    print('Best number of clusters: {}'.format(kl.elbow))
