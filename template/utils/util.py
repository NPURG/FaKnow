import torch
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cos_matrix(matrix1: torch.Tensor, matrix2: torch.Tensor):
    # pairwise 计算matrix1中每行向量与matrix2中所有行向量的余弦相似度
    return torch.from_numpy(cosine_similarity(matrix1.numpy(), matrix2.numpy()))
