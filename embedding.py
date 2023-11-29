import os

import torch
import numpy as np
import pandas as pd
import logging

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from preprocess import get_data

def load_model():

    """
    We use SBERT for making embeddings.
    https://www.sbert.net/index.html#
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def get_embedding(df, embeddings_path='embeddings.npy'):
    
    model = load_model()
    df = get_data()

    # Check if the file exists
    if not os.path.exists(embeddings_path):
        # If the file does not exist, call the get_embedding method
        logging.info("Creating CSV from JSON file.")
        corpus_embeddings = model.encode(df["abstract"], show_progress_bar=True)
        np.save("./embeddings.npy", corpus_embeddings, allow_pickle=True)
    else:
        logging.info("The file already exists.")
        corpus_embeddings = np.load("./embeddings.npy", allow_pickle=True)

    logging.debug(corpus_embeddings.shape)
    
    return corpus_embeddings

def clustering(corpus_embeddings, num_clusters):

    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment


def main():
    pass
