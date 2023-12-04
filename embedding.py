import os

import torch
import numpy as np
import pandas as pd
import logging

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from yellowbrick.cluster import KElbowVisualizer

from preprocess import get_data

def load_model():

    """
    We use SBERT for making embeddings.
    https://www.sbert.net/index.html#
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def get_embedding(df, embeddings_path='embeddings.npy', names_path='names.npy', mode="abstract"):
    model = load_model()
    
    # Check if the embeddings file exists
    if not os.path.exists(embeddings_path) or not os.path.exists('abstract' + embeddings_path):
        logging.info("Creating embeddings.")
        
        if mode == "abstract":
            # Generate embeddings and save them
            corpus_embeddings = model.encode(df["abstract"], show_progress_bar=True)
            np.save('abstract' + embeddings_path, corpus_embeddings, allow_pickle=True)
            names = df["abstract"].tolist()
            np.save('abstract' + names_path, names, allow_pickle=True)
        else:
            corpus_embeddings = model.encode(df["categories"], show_progress_bar=True)
            np.save(embeddings_path, corpus_embeddings, allow_pickle=True)
            names = df["categories"].tolist()
            np.save(names_path, names, allow_pickle=True)
            
    else:
        logging.info("The embeddings file already exists.")
        if mode == "abstract":
            corpus_embeddings = np.load('abstract' + embeddings_path, allow_pickle=True)
            names = np.load('abstract' + names_path, allow_pickle=True)
        else:
            corpus_embeddings = np.load(embeddings_path, allow_pickle=True)
            names = np.load(names_path, allow_pickle=True)
            
    logging.debug(corpus_embeddings.shape) # type: ignore
    
    return corpus_embeddings, names

def clustering(corpus_embeddings, num_clusters):
    
    logging.info("Start KMeans")
    #clustering_model = KMeans(n_clusters=num_clusters)
    #clustering_model.fit(corpus_embeddings)
    #cluster_assignment = clustering_model.labels_
    for i in range(10):
        clustering_model = KMeans()
        visualizer = KElbowVisualizer(clustering_model, k=(10+i*5, 15+i*5+1)) # type: ignore
        visualizer.fit(corpus_embeddings)
        visualizer.show(outpath=f"range_{10+i*5}_to_{15+i*5}.png")

    return #cluster_assignment

def main():
    pass
