import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cos_matrix(matrix1: np.ndarray, matrix2: np.ndarray):
    # pairwise 计算matrix1中每行向量与matrix2中所有行向量的余弦相似度
    return torch.from_numpy(cosine_similarity(matrix1, matrix2))
