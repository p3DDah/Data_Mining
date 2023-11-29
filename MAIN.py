import logging

from preprocess import get_data
from embedding import get_embedding, clustering
from visualize import create_histograms, create_similarity_matrix

LOG_LEVEL = logging.DEBUG

# Set up logging configuration
logging.basicConfig(level=LOG_LEVEL, format='%(levelname)s: %(message)s')

def main():
    df = get_data()

    #corpus_embeddings = get_embedding(df)
    #cluster_assignment = clustering(corpus_embeddings, num_clusters=30)

    create_histograms(1970, 1980, top_k=5)

    category_list = []
    create_similarity_matrix(category_list)

if __name__ == "__main__":
    main()