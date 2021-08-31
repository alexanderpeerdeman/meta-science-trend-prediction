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

import pandas as pd
import json
import os
from collections import OrderedDict
from matplotlib import pyplot as plt
import sklearn
import numpy as np

json_path = "data/semantic_scholar/papers/"
paper_json_files = os.listdir(json_path)

# +
years_to_consider = range(1990, 2022)
citations_without_year = 0

all_papers_with_years = []

for jf in paper_json_files:
    file_name_or_paper_id = os.path.splitext(jf)[0]

    with open(json_path + jf) as filebuffer:
        paper = json.load(filebuffer)
        
    # initialize one bucket per year
    paper_historical_citations = dict()
    for year in years_to_consider:
        paper_historical_citations[year] = 0

    for cit in paper["citations"]:
        if cit["year"] is None:
            # print("[W] No year: {}".format(cit["title"]))
            citations_without_year += 1
            continue
        if cit["year"] > max(years_to_consider):
            print("[W] Citation newer than interval: {}".format(cit["year"]))
        for year in years_to_consider:
            if cit["year"] <= year:
                # year of citation can be earlier than paper publish year
                # --> might be a mistake but unimportant in aggregated view
#                 if paper["year"] is not None and cit["year"] < paper["year"]:
#                     print("Citation < Paper", cit["year"]-paper["year"])

                paper_historical_citations[year] += 1
    
    paper_historical_citations.update({"id": file_name_or_paper_id})
    all_papers_with_years.append(paper_historical_citations)    
print("Citations without year: {}".format(citations_without_year))

# +
#
# same code as above, but does not accumulate historic citation count
#

years_to_consider = range(1990, 2022)
citations_without_year = 0

all_papers_with_years = []

for jf in paper_json_files:
    file_name_or_paper_id = os.path.splitext(jf)[0]

    with open(json_path + jf) as filebuffer:
        paper = json.load(filebuffer)

        # initialize one bucket per year
        paper_historical_citations = dict()
        for year in years_to_consider:
            paper_historical_citations[year] = 0

        for cit in paper["citations"]:
            if cit["year"] is None:
                citations_without_year += 1
                continue

            
            for year in years_to_consider:
                if cit["year"] == year:
                    paper_historical_citations[year] += 1

        paper_historical_citations.update({"id": file_name_or_paper_id})
        all_papers_with_years.append(paper_historical_citations)    
print("Citations without year: {}".format(citations_without_year))
# -

len(all_papers_with_years)

pd.set_option("display.min_rows", 100)
pd.set_option("display.max_columns", None)

df = pd.DataFrame(all_papers_with_years)
df

df.iloc[17][:-1].plot()

# +
plt.figure(figsize=(9, 9))

plt.scatter(df[2019], df[2020])
lim = 4000
plt.ylim(-10, lim)
plt.xlim(-10, lim)

plt.show()
# -

#plt.figure(figsize=(16, 200))
plt.pcolor(df.head(100)[df.columns[:-1]])

author_count = []
for cur in df.itertuples():
    with open("data/semantic_scholar/papers/"+cur.id+".json", "r") as fb:
        author_count.append(len(json.load(fb)["authors"]))

author_count = np.array(author_count).reshape(-1, 1)
author_count

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

np.set_printoptions(suppress=True)

# +
############
# features #
############
# annahme: wir schreiben das jahr current_year -> daten bis ende current_year-1 sind verfügbar.
current_year = 2010

citations_immediate_past = np.array(df[range(current_year-20, current_year)])
print(citations_immediate_past)
print(citations_immediate_past.shape)

features = np.append(citations_immediate_past, author_count, axis=1)
features
# -

# wir wollen die anzahl an citations predicten, die im current_year hinzukommen
targets = np.array(df[range(current_year, current_year+5)])
targets

data = np.append(features, targets, axis=1)
data

train, test = train_test_split(data, train_size=0.8)

# +
train_x = train[:, :-len(targets[1, :])]

train_y = train[:, -len(targets[1, :]):].T

# +
test_x = test[:, :-len(targets[1, :])]

test_y = test[:, -len(targets[1, :]):].T
# -

train_y[0].shape

for year in range(5):
    print("predicting for {}".format(current_year + year))

    lr = LinearRegression(positive=True).fit(train_x, train_y[year])
    print("score training:", lr.score(train_x, train_y[year]))
    
    print("score testing:", lr.score(test_x, test_y[year]))
    predicted = lr.predict(test_x)
    print("Spearman Correlation:", spearmanr(test_y[year], predicted))
    
    max_val = (0, 0, 0, 0) 
    for i, (y_hat, y) in enumerate(zip(predicted, test_y[year])):
        if abs(y_hat-y) > max_val[3]:
            max_val = (i, y, y_hat, y-y_hat)

    print("Maximum distance", max_val)


import keras
from keras.layers import InputLayer, SimpleRNN, Dense, TimeDistributed, Conv1D
import tensorflow as tf

model = keras.Sequential()
#model.add(SimpleRNN(4)(tf.random.normal([32, 10, 8])))
#model.add(Conv1D(filters=4, kernel_size=4))
model.add(Dense(1000, activation='relu', input_shape=[train_x.shape[1]]))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1000, activation='relu'))
#model.add(Dense(20))
#model.add(Dense(10))
model.add(Dense(5))

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="mse")#, metrics="cosine_similarity")

hist = model.fit(train_x, train_y.T, epochs=5, batch_size=10, validation_data=(test_x, test_y.T))

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])

model.predict(test_x)[30:45].round()

test_y.T[30:45]

test_x[30:45]




















