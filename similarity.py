import numpy as np
from scipy.stats import multivariate_normal, multivariate_t
import os
import torch
import pandas as pd
from sklearn.cluster import KMeans
from preprocess import get_csv
from sklearn.metrics.pairwise import cosine_similarity
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def kl_divergence(X0, X1):
    mean0 = np.mean(X0, axis=0)
    cov0 = np.cov(X0, rowvar=False)
    mean1 = np.mean(X1, axis=0)
    cov1 = np.cov(X1, rowvar=False)

    print(np.mean(cov0))
    print(np.max(cov0))
    print(np.mean(cov1))
    print(np.max(cov1))

    mvn0 = multivariate_normal(mean=mean0, cov=cov0, allow_singular = True)
    mvn1 = multivariate_normal(mean=mean1, cov=cov1, allow_singular = True)
    
    # X = mvn0.rvs(size=1000)
    
    pdf0 = mvn0.logpdf(X0)
    pdf1 = mvn1.logpdf(X0)
    
    for i in range(pdf0.shape[0]):
        if pdf0[i] == -np.inf:
            pdf0[i] = -1e5
    for i in range(pdf1.shape[0]):
        if pdf1[i] == -np.inf:
            pdf1[i] = -1e5
    
    kl_div = np.mean(pdf0-pdf1)
    
    return kl_div

def cos_similarity(X0, X1):
    X0 = X0/np.linalg.norm(X0, axis=1)[:, np.newaxis]
    X1 = X1/np.linalg.norm(X1, axis=1)[:, np.newaxis]
    
    cos_sim = cosine_similarity(X0, X1)
    avg_sim = np.mean(cos_sim)
    
    return avg_sim

def create_sim_matrix(sim, title = 'hello', name = 'hi.png', fmt=".0f"):
    folder_name = "Matrix"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Check if the data is a square matrix
    if sim.shape[0] != sim.shape[1]:
        logging.info("Data is not in the correct format. It should be a square matrix.")
        return
    
    # Reverse the order of the columns
    # data = data[data.columns[::-1]]

    plt.figure()
    heatmap = sns.heatmap(sim, annot=True, cmap='PiYG', fmt = fmt)
    plt.title(title)
    plt.tight_layout()

    # Save the matrix to the specified folder
    save_path = os.path.join(folder_name, name)
    heatmap.figure.savefig(save_path)
    logging.info(f"matrix saved to {save_path}")

    # Optionally, display the matrix
    plt.show()
    
def create_year_matrix(sim, title = 'hello', name = 'hi.png', fmt=".0f"):
    folder_name = "Matrix/years"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Check if the data is a square matrix
    if sim.shape[0] != sim.shape[1]:
        logging.info("Data is not in the correct format. It should be a square matrix.")
        return
    
    # Reverse the order of the columns
    # data = data[data.columns[::-1]]
    plt.figure()
    heatmap = sns.heatmap(sim, vmin=0, vmax=1, annot=True, cmap='PiYG', fmt = fmt)
    plt.title(title)
    plt.tight_layout()

    # Save the matrix to the specified folder
    save_path = os.path.join(folder_name, name)
    heatmap.figure.savefig(save_path)
    logging.info(f"matrix saved to {save_path}")

    # Optionally, display the matrix
    plt.show()