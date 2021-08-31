from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
from sentence_transformers import util
import umap
import util as u
import hdbscan

# Kmeans clustering of embeddings
# Returns mapping of clusters to paper's indices in embeddings, 
# the cluster labels as list (ordered like embeddings), and the centers of the clusters
def kmeans(embeddings, num_clusters=40, random_state=None, visualize=False):
    
    model = KMeans(num_clusters, random_state=random_state).fit(preprocessing.normalize(embeddings))
    labels = model.labels_
    
    assert len(labels) == len(embeddings)
    
    cluster2indices = dict()
    for i, label in enumerate(labels):
        if label not in cluster2indices:
            cluster2indices[int(label)] = []
        cluster2indices[int(label)].append(i)
        
    assert len(labels) == sum([len(cluster2indices[x]) for x in cluster2indices])
        
    if visualize:
        u.visualize_embeddings(embeddings, labels)
      
    return cluster2indices, [int(x) for x in labels], list(model.cluster_centers_)

# Agglomerative clustering of embeddings
# State num_clusters or distance threshold between vectors
# Returns mapping of clusters to paper's indices in embeddings and
# the cluster labels as list (ordered like embeddings)
def agglomerative_clustering(embeddings, num_clusters=None, distance_threshold=1.0, visualize=False):
    
    model = AgglomerativeClustering(n_clusters=num_clusters, distance_threshold=distance_threshold, affinity='cosine', linkage='average')
    model.fit(embeddings)
    labels = model.labels_
    
    assert len(labels) == len(embeddings)
    
    cluster2indices = dict()
    for i, label in enumerate(labels):
        if label not in cluster2indices:
            cluster2indices[int(label)] = []
        cluster2indices[int(label)].append(i)
        
    assert len(labels) == sum([len(cluster2indices[x]) for x in cluster2indices])
        
    if visualize:
        u.visualize_embeddings(embeddings, labels)
        
    return cluster2indices, [int(x) for x in labels]

# Community detection clustering of embeddings
# Returns mapping of clusters to paper's indices in embeddings, 
# the cluster labels as list (ordered like embeddings), and the centers of the clusters
def fast_clustering(embeddings, threshold=0.75, min_community_size=1, visualize=False):
    
    clusters = util.community_detection(embeddings, min_community_size=min_community_size, threshold=threshold)
    
    cluster2indices = dict()
    for i, cluster in enumerate(clusters):
        centers.append(int(cluster[0]))
        cluster2indices[i] = [int(x) for x in cluster]
        
    assert len(embeddings) == sum([len(cluster2indices[x]) for x in cluster2indices])
        
    labels = [-1 for x in range(len(embeddings))]
    for cluster in cluster2indices:
        for p in cluster2indices[cluster]:
            labels[p] = cluster
    
    assert len(labels) == len(embeddings)
    
    
    if visualize:
        u.visualize_embeddings(embeddings, labels)
        
    return cluster2indices, [int(x) for x in labels], centers

# HDBSCAN clustering of embeddings using dimension reduction with UMAP
# https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
# Returns mapping of clusters to paper's indices in embeddings and 
# the cluster labels as list (ordered like embeddings)
def topic_clustering(embeddings, n_neighbors=15, n_components=5, min_cluster_size=15, visualize=False):
    
    umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
                            n_components=n_components, 
                            metric='cosine').fit_transform(embeddings)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)
    labels = model.labels_
    
    assert len(labels) == len(embeddings)
    
    cluster2indices = dict()
    for i, label in enumerate(labels):
        if label not in cluster2indices:
            cluster2indices[int(label)] = []
        cluster2indices[int(label)].append(i)
        
    assert len(labels) == sum([len(cluster2indices[x]) for x in cluster2indices])
        
    if visualize:
        u.visualize_embeddings(umap_embeddings, labels)
        
    return cluster2indices, [int(x) for x in labels]
