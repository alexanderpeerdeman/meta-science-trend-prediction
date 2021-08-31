import numpy as np
import itertools
import nltk
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

# Get number of clusters
def get_num_clusters(cluster2indices):
    return len(cluster2indices)

# Get min size of clusters
def get_min_cluster_size(cluster2indices):
    return min([len(cluster2indices[x]) for x in cluster2indices])

# Get max size of clusters
def get_max_cluster_size(cluster2indices):
    return max([len(cluster2indices[x]) for x in cluster2indices])

# Get mean size of clusters
def get_mean_cluster_size(cluster2indices):
    return np.mean([len(cluster2indices[x]) for x in cluster2indices])

# get median size of clusters
def get_median_cluster_size(cluster2indices):
    return np.median([len(cluster2indices[x]) for x in cluster2indices])

# Intra Cluster Similarity - semantic scholar topics or cso topics
# Calculate for each cluster the percentage of papers having at least one keyword in top/most common
# n_ keywords of clusters, then average percentages
# source: 'cso' or 'sem_scholar'
def intra_cluster_similarity(cluster2indices, cluster2words_freq_dist, source, df_clustered, n=[10]):
    
    # for each cluster, how many papers are with at least one keyword represented in top n keywords 
    n2cluster2represented_papers = dict()
    
    # for all n_ in n consider top n_ keywords
    for n_ in n:
        n2cluster2represented_papers[n_] = dict()
        
        # for all clusters 
        for cluster_index in cluster2indices:
            
            # get cluster = list of paper indices in dataframe
            cluster = cluster2indices[cluster_index]
            
            # initialise current number of papers with 0
            n2cluster2represented_papers[n_][cluster_index] = 0
            
            # get top, most common n_ keywords for current cluster 
            most_common = [x for (x, _) in cluster2words_freq_dist[cluster_index].most_common(n_)]
            
            # check for all papers if they have at least one keyword in most_common list
            for paper_index in cluster:
                
                if source == "sem_scholar":
                    # get semantc scholar keywords of current paper
                    words = df_clustered.iloc[paper_index]["semantic_scholar_keywords"]
                elif source == "cso":
                    # get cso topics of current paper
                    words = df_clustered.iloc[paper_index]["cso_enhanced"]
                
                if not set(words).isdisjoint(set(most_common)):
                    
                    # increase number of papers in dictionary with 1
                    n2cluster2represented_papers[n_][cluster_index] += 1
                    
    # for each n_ in n, calculate the mean of the percentages of papers with at least one
    # keyword in top n_ of each cluster
    # i.e. cluster 0: 5/10 have key words in top n_ => 50% = 0.5
    #      cluster 1: 2/10 have key words in top n_ => 25% = 0.2
    #      => average for n_: 0.35
    
    # for all n_ in n, store mean percentage over clusters
    n2mean_percentage = dict()
    
    for n_ in n:
        percentages = []
        
        
        for cluster_index in n2cluster2represented_papers[n_]:
            if int(cluster_index) != -1:
                repr_papers = n2cluster2represented_papers[n_][cluster_index]
                total_papers = len(cluster2indices[cluster_index])
                percentages.append(repr_papers/total_papers)
        n2mean_percentage[n_] = np.mean(percentages)
        
    return n2mean_percentage

# Inter Cluster Similarity - semantic scholar keywords or cso topics
# Calculate for all cluster pairs (c1, c2) the spearman correlation of the lists of top n_ keywords,
# then average these similarity scores over number of pairs
# source: 'cso' or 'sem_scholar'
def inter_cluster_similarity(cluster2indices, cluster2words_freq_dist, source, n=[10]):
    
    cluster_indices = list(cluster2indices.keys())
    if "-1" in cluster_indices:
        cluster_indices.remove("-1")
    if -1 in cluster_indices:
        cluster_indices.remove(-1)
    
    # create all pairs of clusters
    cluster_pairs = [(c1, c2) for c1, c2 in itertools.combinations(cluster_indices, 2)]

    # for each top/most_common n_ keyworsd, store similarity value 
    # (spearman correlation of top n_ keywords lists) of each cluster pair
    n2pair_similarities = dict()
    
    for n_ in n:
        n2pair_similarities[n_] = []
        
        for c1, c2 in cluster_pairs:
        
            words_freq_dist_c1 = cluster2words_freq_dist[c1]
            words_freq_dist_c2 = cluster2words_freq_dist[c2]
            
            c1_n_most_common = [x for x, _ in words_freq_dist_c1.most_common(n_)]
            c2_n_most_common = [x for x, _ in words_freq_dist_c2.most_common(n_)]
            
            # We need two lists with equal size for spearman correlation
            min_length = min(len(c1_n_most_common), len(c2_n_most_common))
        
            correlation, _ = spearmanr(c1_n_most_common[:min_length], c2_n_most_common[:min_length])
            
            n2pair_similarities[n_].append(correlation)
    
    # for each n calculate the mean of the correlations of all cluster pairs
    n2mean_correlation = dict()
    
    for n_ in n:
        correlations = n2pair_similarities[n_]
        n2mean_correlation[n_] = np.mean(correlations)
        
    return n2mean_correlation

# Calculate a similarity score of a word list compared to a word frequency distribution
# normalized by cluster size
def wordlist_freq_dist_similarity(words, word_freq_dist, cluster_size):
    
    score = 0
    
    # for each keyword in keywords
    for word in words:
        for i, (w, frequency) in enumerate(word_freq_dist.most_common()):
            
            # if keyword also in frequency distribution
            if w == word:
                
                # add frequency/cluster_size to the score
                score += frequency/cluster_size
                break
                
    return score

# Calculate 1. the accuracy, 2. a mean ranking score of new unclustered papers, and 3. a absolute score based on % of papers with words
# parameter gold: a list of clusters, one cluster per classificated paper
# source: 'cso' or 'sem_scholar'
def classification_acc_ranking(gold, cluster2words_freq_dist, source, cluster2indices, df_not_clustered):
    
    # for each new classified paper (identified by index in df_not_clustered), get its keywords
    index2words = dict()
    for i, row in df_not_clustered.iterrows(): 
        if source == "sem_scholar":
            index2words[i] = row["semantic_scholar_keywords"]
        elif source == "cso":
            index2words[i] = row["cso_enhanced"]
        else:
            print("Warning. Source not found!")
    
    # for each paper index store mapping from cluster to score
    index2cluster2score = dict()
    
    for index in index2words:
        index2cluster2score[index] = dict()
        words = index2words[index]
        
        # for each cluster
        for cluster in cluster2words_freq_dist:
            
            # get similarity score of paper's words and the word frequency distribution of the current cluster
            score = wordlist_freq_dist_similarity(words, cluster2words_freq_dist[cluster], len(cluster2indices[cluster]))
            
            # save score in dictionary
            index2cluster2score[index][cluster] = score
    
    # 1. first metric - accuracy of correct clusters with regard to gold (classified with classifier)
    # for each paper index store cluster with highest score
    index2best_cluster = dict()
    for index in index2words:
        max_score = -1
        best_cluster = 0
        
        for cluster in index2cluster2score[index]:
            if int(cluster) != -1:
                score = index2cluster2score[index][cluster]
                if score > max_score:
                    max_score = score
                    best_cluster = cluster
        
        index2best_cluster[index] = best_cluster
    
    # create list of cluster predictions based on keywords
    predicted = [index2best_cluster[index] for index in index2best_cluster]
        
    # calculate accuracy
    acc = accuracy_score(gold, predicted)
    
    # 2. second metric - at which position is correct cluster in list of clusters sorted by score
    # list of scores: g/(p * g)
    # g: number of clusters 
    # p: position of gold cluster in sorted score list of paper
    ranking_scores = []
    for index in index2words:
        cluster2score = index2cluster2score[index]
        for i, (cluster, score) in enumerate(sorted(cluster2score.items(), key= lambda x:x[1], reverse=True)):
            if gold[index] == cluster:
                #ranking_scores.append(len(cluster2score)/((i+1) * len(cluster2score)))
                ranking_scores.append(1/(i+1))
                break
                
    # calculate mean ranking score
    mean_ranking_score = np.mean(ranking_scores)
    
    # 3. third metric - how many in % of cluster members have the words
    scores = []
    for index in index2words:
        words = index2words[index]
        score = 0
        for word in words:
            score += cluster2words_freq_dist[gold[index]][word] / len(cluster2indices[gold[index]])
        scores.append(score)
        
    mean_absolute_score = np.mean(scores)
    
    return acc, mean_ranking_score, mean_absolute_score
