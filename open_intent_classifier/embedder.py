from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BAII_SMALL_V1_5 = "BAAI/bge-small-en-v1.5"

class StaticLabelsEmbeddingClassifier:
    def __init__(self, labels: List[str], model_name: str = BAII_SMALL_V1_5):
        self.model = SentenceTransformer(model_name)
        self.embedded_labels = self.model.encode(labels, show_progress_bar=True)
        self.labels = labels

    def top_n(self, text: str, n: int, embeddings: np.array) -> (np.array, np.array):
        text_embedding = self.model.encode([text])
        # for matrix multiplication we need the shape to be NXM MXN vector for example: vectpr 1X384, matrix 384X3
        angles = cosine_similarity(text_embedding, embeddings).squeeze()
        # https://stackoverflow.com/a/6910672
        # ::-1 reverses this list, # -n: top N
        sorted_indices = angles.argsort()[-n:][::-1]
        return sorted_indices, angles[sorted_indices]

    def predict(self, text: str, n=1) -> Tuple[List[str], np.array]:
        top_n_indices, top_n_angles = self.top_n(text, n=n, embeddings=self.embedded_labels)
        top_n_labels = [self.labels[i] for i in top_n_indices]

        return top_n_labels, top_n_angles

