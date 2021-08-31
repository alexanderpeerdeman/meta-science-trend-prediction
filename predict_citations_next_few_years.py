# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import json
import os
import pickle
import random
from pathlib import Path
from typing import NamedTuple

import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Normalization
from keras.losses import MeanAbsoluteError, MeanSquaredError
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from util import (load_best_clustering, plot_year2value,
                  transform_indices_dict_to_id_dict)

YEAR_START = 1990
YEAR_END = 2021  # exclusive
YEAR_CURRENT = 2020
YEARS_TO_PREDICT = 5
SEED = 12


class CitationWindow:
    def __init__(self, abstract_embedding, cit_hist, window_year_offset: int, num_authors: int, future_citations):
        super().__init__()
        self.abstract_embedding = abstract_embedding
        self.citation_history = cit_hist
        self.window_year_offset = window_year_offset
        self.num_authors = num_authors
        self.future_citations = future_citations

    def flatten(self):
        return (
            np.append(
                self.abstract_embedding,
                np.append(
                    self.citation_history,
                    [
                        self.window_year_offset,
                        self.num_authors
                    ]
                ),
            ),
            self.future_citations
        )


def make_baseline_predictions(dataset, years_to_predict, baseline_variant):
    preds = []
    for cw in dataset:
        preds.append(baseline_variant(cw, years_to_predict))
    return np.vstack(preds)


def baseline_constant(cw: CitationWindow, years_to_predict: int):
    """Takes a datapoint and a number of years and returns a vector with the
    most recent citation count at every position."""
    most_recent_citation_count = cw.citation_history[-1]
    return np.full(years_to_predict, most_recent_citation_count)


def baseline_trend(cw: CitationWindow, years_to_predict: int):
    """TODO: docstring + comments"""

    # if the window is on the year of publication, there is nothing we can extract in terms of gradient. Assume 0.
    mean_gradient = 0
    if cw.window_year_offset > 1:
        # determine the gradient of the relevant portion of the historical citation vector.
        # get its mean. This represents the citations that on average are added each year.
        mean_gradient = np.gradient(
            cw.citation_history[-cw.window_year_offset:]).mean()

    most_recent_citation_count = cw.citation_history[-1]
    # fill predictions vector with last entry of historical citation vector
    naive_pred = np.full(years_to_predict,
                         most_recent_citation_count)

    # for all years to predict, add the calculated gradient times offset
    for x in range(years_to_predict):
        naive_pred[x] += (x + 1) * mean_gradient

    naive_pred = naive_pred.round()
    return naive_pred


def load_from_cache(filename):
    with open("cache/{}".format(filename), "r") as f:
        print("loaded {} from cache.".format(filename))
        return json.load(f)


def load_papers(jsons_path):
    paper_json_files = os.listdir(jsons_path)
    random.shuffle(paper_json_files)
    # try to load it from cache
    try:
        all_papers = load_from_cache("all_papers.json")
        abstracts = load_from_cache("abstracts.json")
        return all_papers, abstracts
    except FileNotFoundError:
        all_papers, abstracts = [], []

        # get citations per year
        for paper_json_filename in tqdm(paper_json_files):
            with open(jsons_path + paper_json_filename) as jf:
                paper_json = json.load(jf)
            paper_id = os.path.splitext(paper_json_filename)[0]

            # get the publication year of the current paper.
            # We need this to say how old the paper is later
            # (needs post-processing when transforming into training data)

            if paper_json["year"] is None:
                continue

            publication_year = int(paper_json["year"])
            years_since_publication = range(
                max(YEAR_START, publication_year), YEAR_END)

            acc_cit_count_by_year = citation_count_by_year(
                paper_json, years_since_publication)

            # count number of authors

            # if the paper_json file has 0 authors, we say it has exactly 1 author.
            # This is important for later calculations and a paper cannot have 0 authors anyway.
            num_authors = max(1, len(paper_json["authors"]))

            # add abstract of current paper to list of abstracts so that we can perform bulk embedding later.
            if paper_json["abstract"] is None:
                abstracts.append("")
            else:
                abstracts.append(paper_json["abstract"])

            datapoint = {
                "paper_id": paper_id,
                "title": paper_json["title"],
                "publication_year": publication_year,
                "acc_cit_count_by_year": acc_cit_count_by_year,
                "num_authors": num_authors
            }

            all_papers.append(datapoint)

        # write to cache
        Path("cache").mkdir(parents=True, exist_ok=True)

        with open("cache/all_papers.json", "w") as f:
            json.dump(all_papers, f)

        with open("cache/abstracts.json", "w") as f:
            json.dump(abstracts, f)

        return all_papers, abstracts


def citation_count_by_year(paper, years_since_publication):
    """Takes a paper and a list of every year since its publication and returns a dictionary:

    every year since publication --> the amount of citations the paper had in this year."""

    cit_count_by_year = {
        year: 0 for year in years_since_publication}

    # go through all citations
    for citation in paper["citations"]:
        # if the citation has no year, skip it.
        if citation["year"] is None:
            continue

        # increment the counter of all years <= publication year of the citation.
        # this way we get an accumulated historical citation count.
        for year in years_since_publication:
            if citation["year"] <= year:
                cit_count_by_year[year] += 1
    return cit_count_by_year


def create_abstract_embeddings(abstracts):
    """Takes a list of abstracts (a list of strings) and loads the embeddings from cache. 

    If there is no cache we use sentence transformer's bulk embedding method to calculate them and write them to cache."""

    try:
        with open("cache/abstracts_embeddings.pkl", "rb") as f:
            abstracts_embeddings = pickle.load(f)
            print("loaded embeddings from cache.")
    except:
        embeddings_model = SentenceTransformer(
            'paraphrase-distilroberta-base-v2')
        abstracts_embeddings = np.array(
            embeddings_model.encode(
                abstracts,
                show_progress_bar=True
            ))

        # write to cache
        Path("cache").mkdir(parents=True, exist_ok=True)

        with open("cache/abstracts_embeddings.pkl", "wb") as f:
            pickle.dump(abstracts_embeddings, f)

    return abstracts_embeddings


def split_into_features_and_labels(dataset):
    x, y = [], []

    for cw in dataset:
        features, labels = cw.flatten()
        x.append(features)
        y.append(labels)

    x = np.array(x, dtype="float")
    y = np.array(y, dtype="float")
    return x, y


def print_metrics(model_string, dev_or_test, mae, mse):
    print("###################")
    print("Evaluation on {} set: {}:".format(dev_or_test, model_string))
    print("MeanAbsoluteError:\t{}".format(mae))
    print("MeanSquaredError:\t{}".format(mse))
    print("###################")


def create_dataset(all_papers, abstracts_embeddings):
    """Takes metadata from all papers and their abstract embeddings,
    creates all possible citation windows from each paper 
    and returns them as a list of citation windows."""
    dataset = []

    for index, paper in tqdm(enumerate(all_papers)):
        if (paper["publication_year"] + YEARS_TO_PREDICT) > YEAR_CURRENT:
            # skip all papers that we cant create at least one full citation window for
            continue

        citation_counts = list(paper["acc_cit_count_by_year"].values())
        windows_to_create = len(
            citation_counts) - YEARS_TO_PREDICT

        for offset in range(windows_to_create):
            citation_future, citation_history = citation_future_and_history(
                citation_counts, offset)

            cw = CitationWindow(abstracts_embeddings[index],
                                citation_history,
                                offset,
                                paper["num_authors"],
                                citation_future)

            dataset.append(cw)
    return dataset


def citation_future_and_history(citation_counts, offset):
    history = citation_counts[:offset]
    citation_future = citation_counts[offset:offset + YEARS_TO_PREDICT]

    citation_history = np.zeros(YEAR_END-YEAR_START)
    # overwrite the right hand side of the array
    if len(history) > 0:
        citation_history[-len(history):] = history
    return (citation_future, citation_history)


def create_dataset_prediction(all_papers, abstracts_embeddings):
    """Takes the metadata of all papers and their abstract embeddings,
    creates one citation window each, with a papers' age as offset."""
    dataset_for_prediction = []

    for index, paper in tqdm(enumerate(all_papers)):
        # build datapoint as tuple of features only
        citation_counts = list(paper["acc_cit_count_by_year"].values())

        _, citation_history = citation_future_and_history(
            citation_counts, len(citation_counts))

        cw = CitationWindow(abstracts_embeddings[index],
                            citation_history,
                            YEAR_CURRENT - paper["publication_year"],
                            paper["num_authors"],
                            None)

        dataset_for_prediction.append(cw)

    return dataset_for_prediction


def plot_loss(hist):
    plt.plot(hist.history["loss"], label="loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.legend(loc="upper right")
    plt.title("loss and val_loss")
    plt.xlabel("epochs")
    plt.ylabel("Mean Absolute Error")
    plt.savefig("model/model_loss_plot.png")
    plt.show()


### Start ###

jsons_path = "data/semantic_scholar/papers/"
all_papers, abstracts = load_papers(jsons_path)

abstracts_embeddings = create_abstract_embeddings(abstracts)


try:
    model = keras.models.load_model("model/best_model.hdf5")
    print("Loaded model from cache.")
except:
    assert len(abstracts_embeddings) == len(all_papers)
    np.random.seed(SEED)

    dataset = create_dataset(all_papers, abstracts_embeddings)

    # create train/dev/test split
    train_and_validation, test = train_test_split(
        dataset, test_size=0.2, random_state=SEED)
    train, validation = train_test_split(
        train_and_validation, test_size=0.2, random_state=SEED)

    train_x, train_y = split_into_features_and_labels(train)
    validation_x, validation_y = split_into_features_and_labels(validation)
    test_x, test_y = split_into_features_and_labels(test)

    # constant baseline
    baseline_pred = make_baseline_predictions(
        validation, YEARS_TO_PREDICT, baseline_constant)

    # evaluate constant baseline: MAE & MSE
    baseline_loss_mae = MeanAbsoluteError()(validation_y, baseline_pred).numpy()
    baseline_loss_mse = MeanSquaredError()(validation_y, baseline_pred).numpy()
    print_metrics("Baseline (Contant)", "dev",
                  baseline_loss_mae, baseline_loss_mse)

    # trend baseline
    baseline_pred_2 = make_baseline_predictions(
        validation, YEARS_TO_PREDICT, baseline_trend)

    # evaluate trend baseline: MAE & MSE
    baseline_loss_mae = MeanAbsoluteError()(validation_y, baseline_pred_2).numpy()
    baseline_loss_mse = MeanSquaredError()(validation_y, baseline_pred_2).numpy()
    print_metrics("Baseline (Trend)", "dev",
                  baseline_loss_mae, baseline_loss_mse)

    # build model
    normalizer = Normalization(input_shape=[train_x.shape[1]])
    normalizer.adapt(train_x)

    model = keras.Sequential([
        normalizer,
        Dense(units=32, activation="relu"),
        Dropout(0.05),
        Dense(units=16, activation="relu"),
        Dropout(0.01),
        Dense(units=YEARS_TO_PREDICT, activation="relu")
    ])

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="mae", metrics=["mse"])
    model.summary()

    # callbacks
    es = EarlyStopping(monitor="val_loss",
                       verbose=1,
                       patience=8)
    mc = ModelCheckpoint("model/best_model.hdf5",
                         save_best_only=True,
                         monitor="val_loss",
                         verbose=1)

    # train model
    hist = model.fit(
        train_x,
        train_y,
        epochs=500,
        batch_size=16,
        validation_data=(validation_x, validation_y),
        callbacks=[mc, es]
    )

    print("##### Training done. #####")

    # evaluate model
    test_set_metrics = model.evaluate(test_x, test_y)
    print_metrics("Ours", "test", test_set_metrics[0], test_set_metrics[1])

    print("loss:", hist.history["loss"])
    print("val_loss:", hist.history["val_loss"])

    plot_loss(hist)

    print("Showing 15 sample predictions:")
    print("predicted:")
    print(model.predict(test_x)[:15].round())
    print("actual:")
    print(test_y[:15])


# prepare dataset to predict every papers future citations
dataset_for_prediction = create_dataset_prediction(
    all_papers, abstracts_embeddings)
inference_set, _ = split_into_features_and_labels(dataset_for_prediction)
predicted = model.predict(inference_set)

# load the clusterings
clustering_dict, _, _ = load_best_clustering()
clustering_dict = transform_indices_dict_to_id_dict(clustering_dict)

# for every cluster, get the respective citation count predictions, sum and visualize them
future_citations_by_cluster_index = dict()
max_sum_of_citation_counts = 0
for index_cluster, paper_ids in clustering_dict.items():
    future_citations_by_year = dict()

    # get the citation counts from all the papers of the current cluster
    citation_counts = []
    for index_paper, paper in enumerate(all_papers):
        if paper["paper_id"] + ".json" in paper_ids:
            citation_counts.append(predicted[index_paper])
            continue

    # sum them up
    future_citations = np.stack(citation_counts, axis=0).sum(axis=0)

    # transform dict keys back from offsets to acutal years
    for i in range(YEARS_TO_PREDICT):
        year = YEAR_CURRENT + i + 1
        future_citations_by_year[year] = future_citations[i]
        max_sum_of_citation_counts = max(max_sum_of_citation_counts, future_citations[i])

    future_citations_by_cluster_index[index_cluster] = future_citations_by_year

# show a plot for each cluster
for cluster in future_citations_by_cluster_index.keys():
    plot_year2value(future_citations_by_cluster_index, cluster, "# citations",
                    max_sum_of_citation_counts)
