import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
params = {'axes.titlesize':'14',
          'xtick.labelsize':'14',
          'ytick.labelsize':'14'}
matplotlib.rcParams.update(params)
import nltk
import umap
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# Split full anthology dataframe with respect to conferences
# Return dictionary
def split_to_conferences(df):
    df_acl   = df[df['url'].str.match(r'(.*anthology/20..\.acl-main.*)') 
                | df['url'].str.match(r'(.*anthology/P..-1.*)')]
    df_emnlp = df[df['url'].str.match(r'(.*anthology/20..\.emnlp-main.*)') 
                | df['url'].str.match(r'(.*anthology/D0[7-9]-1.*)')
                | df['url'].str.match(r'(.*anthology/D1.-1.*)')
                | df['url'].str.match(r'(.*anthology/W06-16[0-7].*)')
                | df['url'].str.match(r'(.*anthology/H05-1[0-1].*)')
                | df['url'].str.match(r'(.*anthology/W04-32.*)')
                | df['url'].str.match(r'(.*anthology/W03-10.*)')
                | df['url'].str.match(r'(.*anthology/W02-10.*)')
                | df['url'].str.match(r'(.*anthology/W01-05.*)')
                | df['url'].str.match(r'(.*anthology/W00-13.*)')
                | df['url'].str.match(r'(.*anthology/W99-06.*)')
                | df['url'].str.match(r'(.*anthology/W98-15[0-1].*)')
                | df['url'].str.match(r'(.*anthology/W97-03[0-2].*)')
                | df['url'].str.match(r'(.*anthology/W96-02[0-1].*)')]
    df_naacl = df[df['url'].str.match(r'(.*anthology/N1[0|2|3|5|6|8|9]-1.*)', na=False)
                | df['url'].str.match(r'(.*anthology/N0[1|3|4|6|7|9]-1.*)', na=False)
                | df['url'].str.match(r'(.*anthology/A00-1.*)', na=False)
                | df['url'].str.match(r'(.*anthology/A00-2.*)', na=False)]
    df_coling = df[df['url'].str.match(r'(.*anthology/20..\.coling-main.*)') 
                | df['url'].str.match(r'(.*anthology/C1[0|2|4|6|8]-1.*)')
                | df['url'].str.match(r'(.*anthology/C0[0|2|4|8]-1.*)')
                | df['url'].str.match(r'(.*anthology/C9[0|2|4|6|8]-1.*)')
                | df['url'].str.match(r'(.*anthology/C9[0|2|4|6|8]-2.*)')
                | df['url'].str.match(r'(.*anthology/C9[0|2]-3.*)')
                | df['url'].str.match(r'(.*anthology/C92-4.*)')]
    df_conll = df[df['url'].str.match(r'(.*anthology/20..\.conll-1.*)') 
                | df['url'].str.match(r'(.*anthology/K1[5-9]-1.*)')
                | df['url'].str.match(r'(.*anthology/W14-16.*)')
                | df['url'].str.match(r'(.*anthology/W13-35.*)')
                | df['url'].str.match(r'(.*anthology/W11-03.*)')
                | df['url'].str.match(r'(.*anthology/W10-29.*)')
                | df['url'].str.match(r'(.*anthology/W09-11.*)')
                | df['url'].str.match(r'(.*anthology/W08-21.*)')
                | df['url'].str.match(r'(.*anthology/W06-29.*)')
                | df['url'].str.match(r'(.*anthology/W05-06.*)')
                | df['url'].str.match(r'(.*anthology/W04-24.*)')
                | df['url'].str.match(r'(.*anthology/W03-04.*)')
                | df['url'].str.match(r'(.*anthology/W02-20.*)')
                | df['url'].str.match(r'(.*anthology/W01-07.*)')
                | df['url'].str.match(r'(.*anthology/W00-07.*)')
                | df['url'].str.match(r'(.*anthology/W99-07.*)')
                | df['url'].str.match(r'(.*anthology/W98-12.*)')
                | df['url'].str.match(r'(.*anthology/W97-10.*)')]
    df_tacl = df[df['url'].str.match(r'(.*anthology/20..\.tacl-1.*)') 
                | df['url'].str.match(r'(.*anthology/Q1[3-9]-1.*)')]
    
    conf2df = {"acl":df_acl, "emnlp": df_emnlp, "naacl": df_naacl, "coling": df_coling, "conll": df_conll, "tacl": df_tacl}

    return conf2df


# Save each conference dataframe into csv file
def save_conf2df(conf2df):
    conf2df["acl"].to_csv("data/anthology_acl.csv", sep="|", index=False)
    conf2df["emnlp"].to_csv("data/anthology_emnlp.csv", sep="|", index=False)
    conf2df["naacl"].to_csv("data/anthology_naacl.csv", sep="|", index=False)
    conf2df["coling"].to_csv("data/anthology_coling.csv", sep="|", index=False)
    conf2df["conll"].to_csv("data/anthology_conll.csv", sep="|", index=False)
    conf2df["tacl"].to_csv("data/anthology_tacl.csv", sep="|", index=False)


# Load embeddings from file
def load_embeddings(filename):
    with open('data/embeddings/' + filename, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_texts = stored_data['texts']
        stored_embeddings = stored_data['embeddings']

    return stored_texts, stored_embeddings

# Visualize embeddings in reduced 2D vector space 
# labels: list of cluster labels, first value is for first embedding, etc.
# cluster: True for colored separation of clusters, else False
def visualize_embeddings(embeddings, labels, cluster=True):
    # Prepare data into 2D vector space
    umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = np.array(labels)

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(16, 9))
    outliers = result.loc[np.array(labels) == -1, :]
    clustered = result.loc[np.array(labels) != -1, :]
    
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    if cluster:
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    else:
        plt.scatter(clustered.x, clustered.y, color='#BDBDBD', s=0.05)
    plt.colorbar()
    plt.savefig("figures/embeddings_in_2D.pdf")


# Plot and stores the histogram with number of papers per cluster
def plot_cluster_hist(cluster2indices):
    x = sorted([int(a) for a in cluster2indices.keys()])
    y = [len(cluster2indices[str(c)]) for c in x]
    
    plt.bar([str(a) for a in x], y)
    plt.xlabel("cluster", fontsize=14)
    plt.ylabel("# papers", fontsize=14)
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.savefig("figures/cluster_dist.pdf")
    plt.show()


# Create classifier training set for one clustering as well as the cluster2indices mapping and the test set
def get_classifier_data_sets(config2clusters, df, last_year, embeddings, pretrained_model, text_set, algorithm, num_clusters, distance_thresholds, min_cluster_size, neighbors, components):
    
    d = config2clusters[pretrained_model][text_set][algorithm]["num_clusters"][str(num_clusters)]["distance_thresholds"][str(distance_thresholds)]["min_cluster_sizes"][str(min_cluster_size)]["neighbors"][str(neighbors)]["components"][str(components)]
    cluster2indices = d["cluster2indices"]
    
    num_clustered_papers = sum([len(cluster2indices[x]) for x in cluster2indices])
    
    #print(num_clustered_papers)
    
    X_train = []
    for i, row in df.iterrows():
        if row["year"] <= last_year:
            if -1 in cluster2indices and i in cluster2indices[-1]:
                continue
            X_train.append(embeddings[i])
            
    assert num_clustered_papers == len(X_train), "{} vs {}".format(num_clustered_papers, len(X_train))
    
    y_train = [-2 for _ in range(len(X_train))]
    for cluster in cluster2indices:
        if cluster != -1:
            indices = cluster2indices[cluster]
            for index in indices:
                y_train[index] = cluster

    assert len(y_train) == len(X_train), "{} vs {}".format(len(y_train), len(X_train))
    assert len([1 for x in y_train if x == -2]) == 0, len([1 for x in y_train if x == -2])
    assert -1 not in y_train
    
    X_predict = []
    for i, row in df.iterrows():
        if row["year"] > last_year:
            X_predict.append(embeddings[i])
            
    return cluster2indices, X_train, y_train, X_predict


# Create classifier training set for one clustering as well as the cluster2indices mapping and the test set
def get_classifier_data_sets_2(cluster2indices, df, last_year, embeddings, pretrained_model, text_set, algorithm, num_clusters, distance_thresholds, min_cluster_size, neighbors, components):
    
    num_clustered_papers = sum([len(cluster2indices[x]) if x != -1 else 0 for x in cluster2indices])
    
    X_train = []
    for i, row in df.iterrows():
        if row["year"] <= last_year:
            if -1 in cluster2indices and i in cluster2indices[-1]:
                continue
            X_train.append(embeddings[i])
            
    assert num_clustered_papers == len(X_train), "{} vs {}".format(num_clustered_papers, len(X_train))
    
    y_train = [-2 for _ in range(len(X_train))]
    for cluster in cluster2indices:
        if cluster != -1:
            indices = cluster2indices[cluster]
            for index in indices:
                y_train[index] = cluster

    assert len(y_train) == len(X_train), "{} vs {}".format(len(y_train), len(X_train))
    assert len([1 for x in y_train if x == -2]) == 0, len([1 for x in y_train if x == -2])
    assert -1 not in y_train
    
    X_predict = []
    for i, row in df.iterrows():
        if row["year"] > last_year:
            X_predict.append(embeddings[i])
            
    return X_train, y_train, X_predict


# +
# Map each cluster to its top n semantic scholar topics / cso topics
# source: 'cso' or 'sem_scholar'
def get_cluster2words(cluster2indices, source, df_clustered, n=10):

    cluster2words = dict()

    # iterate over all clusters
    for cluster_index in cluster2indices:

        # get cluster = list of paper indices in dataframe
        cluster = cluster2indices[cluster_index]

        # store all words
        words = []

        # for all papers in current cluster
        for paper_index in cluster:

            # extend word list with words of paper at paper index in dataframe with only clustered papers
            if source == "sem_scholar":
                words.extend(
                    df_clustered.iloc[paper_index]["semantic_scholar_keywords"])
            elif source == "cso":
                words.extend(df_clustered.iloc[paper_index]["cso_enhanced"])
                # Select only nlp topics
                # # TODO?
            else:
                print("Warning. Source not found!")

        # for current cluster create frequency distriburion over keywords
        cluster2words[cluster_index] = [word for word,
                                        _ in nltk.FreqDist(words).most_common(n)]

    return cluster2words

# +
# Map each cluster to its top n semantic scholar topics / cso topics
# source: 'cso' or 'sem_scholar'
def get_cluster2words_freq_dist(cluster2indices, source, df_clustered):
    
    cluster2words = dict()
    
    # iterate over all clusters
    for cluster_index in cluster2indices:
        
        # get cluster = list of paper indices in dataframe
        cluster = cluster2indices[cluster_index]
        
        # store all words
        words = []
        
        # for all papers in current cluster
        for paper_index in cluster:
            
            # extend word list with words of paper at paper index in dataframe with only clustered papers
            if source == "sem_scholar":
                words.extend(df_clustered.iloc[paper_index]["semantic_scholar_keywords"])
            elif source == "cso":
                words.extend(df_clustered.iloc[paper_index]["cso_enhanced"])
                # Select only nlp topics
                # # TODO?
            else:
                print("Warning. Source not found!")
        
        # for current cluster create frequency distriburion over keywords
        cluster2words[cluster_index] = nltk.FreqDist(words)
        
    return cluster2words


# -

# Map each cluster to a dictionary with years as keys and num papers as value
def get_cluster2year2papers(cluster2indices, df_clustered, last_year, normalization=False):

    cluster2year2papers = dict()

    if normalization:
        year2papers = dict()
        for year in range(1990, last_year+1):
            year2papers[year] = len(df_clustered[df_clustered["year"] == year])

    # iterate over all clusters
    for cluster_index in cluster2indices:

        # get cluster = list of paper indices in dataframe
        cluster = cluster2indices[cluster_index]

        # store all years with their num of papers
        year2papers_cluster = dict()
        for year in range(1990, last_year+1):
            year2papers_cluster[year] = 0

        # for all papers in current cluster
        for paper_index in cluster:

            year = df_clustered.iloc[paper_index]["year"]
            year2papers_cluster[year] += 1

        if normalization:
            for year in range(1990, last_year+1):
                year2papers_cluster[year] = year2papers_cluster[year] / \
                    year2papers[year]

        cluster2year2papers[cluster_index] = year2papers_cluster

    return cluster2year2papers


# Map each cluster to a dictionary with years as keys and num citations of its paper in respective year as value
def get_cluster2year2citations(cluster2indices, df_clustered, last_year, normalization=False):

    cluster2year2citations = dict()

    # iterate over all clusters
    for cluster_index in cluster2indices:

        # get cluster = list of paper indices in dataframe
        cluster = cluster2indices[cluster_index]

        # store all years with their num of papers
        year2citations_cluster = dict()
        for year in range(1990, last_year+1):
            year2citations_cluster[year] = 0

        # for all papers in current cluster
        for paper_index in cluster:

            sem_scholar = df_clustered.iloc[paper_index]["semantic_scholar"]
            if sem_scholar != "":
                with open("data/semantic_scholar/papers/" + sem_scholar) as jf:
                    paper = json.load(jf)
                for citation in paper["citations"]:
                    if citation["year"] != None and int(citation["year"]) in year2citations_cluster:
                        year2citations_cluster[int(citation["year"])] += 1

        cluster2year2citations[cluster_index] = year2citations_cluster

    if normalization:
        for year in range(1990, last_year+1):
            citations = 0
            for cluster_index in cluster2indices:
                citations += cluster2year2citations[cluster_index][year]
            for cluster_index in cluster2indices:
                cluster2year2citations[cluster_index][year] = cluster2year2citations[cluster_index][year]/citations

    return cluster2year2citations


# Plot for each year 1. the number of papers or 2. the number of citations for given cluster
def plot_year2value(cluster2year2value, cluster_index, y_label="count", y_lim_top=None, accumulated=False):
    x = []
    y = []
    for year, papers in sorted(cluster2year2value[cluster_index].items(), key=lambda x: x[0]):
        x.append(year)
        if accumulated:
            if len(y) > 0:
                y.append(papers + y[-1])
            else:
                y.append(papers)
        else:
            y.append(papers)

    m, b = np.polyfit(x, y, 1)

    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(np.array(x).reshape(-1, 1), y)
    x_reg = np.array(x)
    y_reg = model.predict(x_reg.reshape(-1, 1))

    plt.xlabel('year')
    plt.ylabel(y_label)
    if y_lim_top is not None:
        plt.ylim(top=y_lim_top*1.1)

    plt.bar(x, y)
    plt.plot(x, m*np.array(x)+b, color="red")
    plt.plot(x_reg, y_reg, color="green")
    plt.title(cluster_index)
    plt.savefig("figures/clusters/{:02d}_year2{}_both.pdf".format(int(cluster_index), y_label[2:]))
    plt.show()


# Plot for each year 1. the number of papers or 2. the number of citations for the given clusters
def plot_year2value_multiple_clusters(cluster2year2value,
                       cluster_indices=[], y_label="count", 
                       y_lim_top=None, accumulated=False):
    
    y_values = []
    x = []
    x_full = False
    for index in cluster_indices:
        y = []
        for year, value in sorted(cluster2year2value[index].items(), key=lambda x: x[0]):
            if not x_full:
                x.append(year)
            if accumulated:
                if len(y) > 0:
                    y.append(value + y[-1])
                else:
                    y.append(value)
            else:
                y.append(value)
        x_full = True
        y_values.append(y)

    X_axis = np.arange(len(x))
    
    #plt.figure(figsize=(16,9))   
    plt.xlabel('year', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tight_layout()
    if y_lim_top is not None:
        plt.ylim(top=y_lim_top*1.1)
    
    for i, y in enumerate(y_values):
        plt.plot(x, y, marker="o", label="Cluster " + cluster_indices[i])
    
    plt.xticks(x)
    plt.legend()
    a = ""
    for x in cluster_indices:
        a = a + x.strip("'") + "_"
    plt.savefig("figures/clusters/multiple_{}year2{}_both.pdf".format(a, y_label[2:]))
    plt.show()


# Plot for each year 1. the number of papers and 2. the number of citations for a given cluster
def plot_year2value_2y(cluster2year2value_1, cluster2year2value_2, 
                       cluster_index, y_label_1="count", y_label_2="count", 
                       y_lim_top_1=None, y_lim_top_2=None, accumulated=False):
    x = []
    y_1 = []
    for year, value in sorted(cluster2year2value_1[cluster_index].items(), key=lambda x: x[0]):
        x.append(year)
        if accumulated:
            if len(y_1) > 0:
                y_1.append(value + y[-1])
            else:
                y_1.append(value)
        else:
            y_1.append(value)
            
    y_2 = []
    for year, value in sorted(cluster2year2value_2[cluster_index].items(), key=lambda x: x[0]):
        if accumulated:
            if len(y_2) > 0:
                y_2.append(value + y[-1])
            else:
                y_2.append(value*100)
        else:
            y_2.append(value*100)

    m_1, b_1 = np.polyfit(x, y_1, 1)
    
    m_2, b_2 = np.polyfit(x, y_2, 1)

    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(np.array(x).reshape(-1, 1), y_1)
    x_reg = np.array(x)
    
    y_1_reg = model.predict(x_reg.reshape(-1, 1))
    model.fit(np.array(x).reshape(-1, 1), y_2)
    y_2_reg = model.predict(x_reg.reshape(-1, 1))

    X_axis = np.arange(len(x))
    
    plt.figure(figsize=(16,9))
    #plt.tight_layout()
    fig,ax = plt.subplots()
    ax.set_xlabel('year', fontsize=14)
    ax.set_ylabel(y_label_1, color="blue", fontsize=14)
    if y_lim_top_1 is not None:
        ax.set_ylim(top=y_lim_top_1*1.1)
    ax.plot(x, y_1, marker="o", color="blue")

    ax2=ax.twinx()
    ax2.set_ylabel(y_label_2, color="red", fontsize=14)
    if y_lim_top_2 is not None:
        ax2.set_ylim(top=y_lim_top_2*100*1.1)
    ax2.plot(x, y_2, color="red", marker="o")
    
    plt.title("Cluster " + cluster_index)
    plt.savefig("figures/clusters/{:02d}_year2{}_both.pdf".format(int(cluster_index), y_label_1[2:]), bbox_inches='tight')
    plt.show()


# Get for one cluster a sorted list (by frequency) of semantic scholar topics
def keywords_from_ids(cluster2keywords, cluster_index):
    # Load mappping from semantic scholar topic ids to topics
    with open("data/semantic_scholar/topicId_mapping.json") as jf:
        id2topic_sem_scholar = json.load(jf)

    keywords = []
    for keyword_id in cluster2keywords[cluster_index]:
        if keyword_id in id2topic_sem_scholar:
            keywords.append(id2topic_sem_scholar[keyword_id])

    return keywords

# Get a mapping with clusters mapped to topic frequency distribution
def ids_freq_dist_to_keyword_freq_dist(cluster2ids_freq_dist):
    # Load mappping from semantic scholar key word ids to key words
    with open("data/semantic_scholar/topicId_mapping.json") as jf:
        id2topic_sem_scholar = json.load(jf)
        
    cluster2keywords_freq_dist = dict()
    for cluster_index in cluster2ids_freq_dist:
        cluster2keywords_freq_dist[cluster_index] = nltk.FreqDist()
        for keyword_id in cluster2ids_freq_dist[cluster_index]:
            if keyword_id in id2topic_sem_scholar:
                cluster2keywords_freq_dist[cluster_index][id2topic_sem_scholar[keyword_id]] = cluster2ids_freq_dist[cluster_index][keyword_id]
    
    return cluster2keywords_freq_dist

# Get topics for one cluster
def topics(cluster2topics, cluster_index):
    return cluster2topics[cluster_index]


# Load best clustering 
def load_best_clustering():
    with open("data/clusters/final_best_one_clustering.json") as jf:
        best = json.load(jf)
    return best["cluster2indices"], best["labels"], best["centers"]


# Compute a score describing how a search word match with a frequency distribution
def topic_cluster_scores(to_search, words_freq_dist, cluster_size):
    score = 0.0
    
    for ts in to_search:
        for word in words_freq_dist:
            if ts.lower() in word.lower():
                score += words_freq_dist[word] / cluster_size
   
    return score


# Searches for the best 3 clusters according to postition of topic in topic frequency distribution
def search_topic(cluster2indices, df, topics, results=3):
    best_clusters_sem_scholar = [(-1, -1.0) for _ in range(results)]
    best_clusters_cso = [(-1, -1.0) for _ in range(results)]
    
    # semantic scholar
    cluster2keywords = get_cluster2words_freq_dist(cluster2indices, "sem_scholar", df)
    cluster2keywords = ids_freq_dist_to_keyword_freq_dist(cluster2keywords)
        
    # cso topics
    cluster2cso = get_cluster2words_freq_dist(cluster2indices, "cso", df)
    
    for cluster_index in cluster2indices:

        # semantic scholar
        sem_scholar_score = topic_cluster_scores(topics, cluster2keywords[cluster_index], len(cluster2indices[cluster_index]))

        for i in range(len(best_clusters_sem_scholar)):
            if sem_scholar_score > best_clusters_sem_scholar[i][1]:
                new_list = []
                for j in range(i):
                    new_list.append(best_clusters_sem_scholar[j])
                new_list.append((cluster_index, sem_scholar_score))
                for j in range(i, len(best_clusters_sem_scholar)):
                    new_list.append(best_clusters_sem_scholar[j])
                best_clusters_sem_scholar = new_list[:results] 
                break
        
        # cso topics
        cso_enhanced_score = topic_cluster_scores(topics, cluster2cso[cluster_index], len(cluster2indices[cluster_index]))
        
        for i in range(len(best_clusters_cso)):
            if cso_enhanced_score > best_clusters_cso[i][1]:
                new_list = []
                for j in range(i):
                    new_list.append(best_clusters_cso[j])
                new_list.append((cluster_index, cso_enhanced_score))
                for j in range(i, len(best_clusters_cso)):
                    new_list.append(best_clusters_cso[j])
                best_clusters_cso = new_list[:results] 
                break
        
    return best_clusters_sem_scholar, best_clusters_cso


def transform_indices_dict_to_id_dict(indicies_dict):
    df = pd.read_csv("data/anthology_conferences.csv", sep="|", keep_default_na=False,
                     converters={"semantic_scholar_keywords": lambda x: x.strip("[]").replace("'", "").split(", "),
                                 "cso_syntactic": lambda x: x.strip("[]").replace("'", "").split(", "),
                                 "cso_semantic": lambda x: x.strip("[]").replace("'", "").split(", "),
                                 "cso_union": lambda x: x.strip("[]").replace("'", "").split(", "),
                                 "cso_enhanced": lambda x: x.strip("[]").replace("'", "").split(", ")})
    id_dict = dict()
    for cluster_index, paper_indicies_this_cluster in indicies_dict.items():
        id_dict[cluster_index] = list(map(lambda x: df["semantic_scholar"][x], paper_indicies_this_cluster))
    return id_dict


def get_history_of_paper_as_fixed_vector(paper, year_offset, desired_length=10):
    citation_history = np.zeros(desired_length)
    counts_by_year = paper["acc_cit_count_by_year"]
    years = counts_by_year.keys()

    for index, year in enumerate(reversed(sorted(years)[:len(years)-year_offset])):
        count = counts_by_year[year]
        citation_history[desired_length-index-1] = count

    return citation_history
