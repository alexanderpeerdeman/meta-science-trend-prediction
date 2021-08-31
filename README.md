# Meta-Science & Evaluation: Trend Prediction

This repository contains the code underlying our term paper **Meta-Science & Evaluation: Trend Prediction**.

### Abstract
The volume of scientific work and related publications has increased sharply in the last two decades. Floods of text can make it even more difficult for academia and industry to evaluate the quality and importance of individual papers and participating in this field of research. 
The ever-advancing power of NLP can help us to process these texts more efficiently than ever before. With this work, we contribute to the ongoing research of metascience and scientometric analysis. In a first step, we derive text embeddings to create unsupervised topic clusters of recent publications and use their citations counts to train a DNN that afterward will forecast relevant topic spaces for new or future research.


## Environment setup

Create new virtual environment:

```bash
$ python -m venv venv
```

Activate environment:

```bash
$ source venv/bin/activate
```

Install required python packages:

```bash
$ pip install -r requirements.txt
```

To view Jupyter Notebooks (`.ipynb`), run:

```bash
$ jupyter-notebook
```

## Download and extract datasets

Download from [https://hessenbox.tu-darmstadt.de/getlink/fiB2mrQTRZTrjWcmCVySL58H/](https://hessenbox.tu-darmstadt.de/getlink/fiB2mrQTRZTrjWcmCVySL58H/).

### Guidelines for extraction and data folder structuring:

We provide a full data set with all results for papers of the ACL anthology from 1990 until 2020.

1. Extract _data.zip_ into the project folder. It includes the complete ACL anthology as well as the filtered anthology according to used years and conferences with all additional information as for example links to files and topics (_anthology_conferences.csv_).
2. Extract _pdfs.zip_ into the new _data/_ folder to add the full paper pdfs.
3. Extract _json.zip_ into the _data/_ folder to add the papers parsed with [science-parse](https://github.com/allenai/science-parse) structured in JSON format.
4. Extract _embeddings.zip_ into the _data/_ folder to add all tested embeddings created with sentenceBERT and different pretrained models.
5. Extract _semantic_scholar.zip_ into the _data/_ folder to add information about papers and authors fetched from _Semantic Scholar_.
6. Extract _clusters.zip_ into the _data/_ folder to add the intermediate and final clustering results. Our best and final clustering is saved into _final_best_onde_clustering.json_.

## Data collection / processing
The basis of the data is [ACL Anthology](https://aclanthology.org/). We further use additional sources to add information e.g. about topics of papers to this basis. Run the following scripts:

1. `parse_data.ipynb` Downloads and filters anthology, downloads paper's pdf, and adds abstracts from parsed pdfs (see next point).
2. `parse_pdf.sh` Parses paper's pdfs with [science-parse](https://github.com/allenai/science-parse) by allenai to get the abstracts.
3. `parse_semanticscholar.ipynb` Downloads [Semantic Scholar](https://semanticscholar.org/) information about paper and authors, and adds topics to anthology entries.
4. `parse_cso_classifier.ipynb` Add topics to each anthology entry based on the abstract using the [python library](https://github.com/angelosalatino/cso-classifier) of the [CSO classifier](https://cso.kmi.open.ac.uk).

## Embedding creation
`embeddings.ipynb` Creates embeddings of titles and abstracts of the papers with SentenceBERT used for clustering.

## Clustering
We use clustering based on embeddings to group papers that share topics. The following steps describe our approach of finding the most appropriate algorithm. Finally, we use K-Means clustering with 20 clusters. The embeddings base on the pretrained model `paraphrase-distilroberta-base-v2` with titles as input.

1. `clustering.ipynb` Runs an extensive search on different clustering algorithms (see `clustering_algorithms.py`), runs an extensive search on filtered algorithm/configurations in `clustering_evaluation.ipynb`, and runs the final best clustering.
2. `clustering_evaluation.ipynb` Manually filters best algorithm/configuration pairst after first and second extensive search using evaluation metrics defined in `clustering_metrics.py`.
3. `cluster_presentation.ipynb` Here you can search for the most matching clusters of keywords/topics and create plots for the clusters regarding the development of citations and papers in the past and the predicted future by the DNN. Figures are stored in the folder _figures/_.

## Model creation

The model is able to predict the citation count of given papers for the next five years, taking

- the embedding (SentenceBERT, `paraphrase-distilroberta-base-v2`)
- the age of the paper since publication
- the accumulated h-indices of all authors and
- the numer of authors.

### Requirements

To create the model,

```
data/semantic_scholar/papers/
```

must exist and contain one JSON file for each paper. Also, to assign the predictions per paper to the cluster,

```
data/clusters/final_best_one_clustering.json
```

has to contain a mapping from cluster index to paper index.

### Train the model and perform predictions

Run:

```bash
$ python predict_citations_next_few_years.py
```

This program writes intermediate results to the `cache/` folder and the resulting model (`keras.callbacks.ModelCheckpoint`) to `model/best_model.hdf5`, alongside a figure showing the development of training and development loss values during the training process.