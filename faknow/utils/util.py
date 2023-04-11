import datetime

import torch
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cos_matrix(matrix1: torch.Tensor, matrix2: torch.Tensor):
    # pairwise 计算matrix1中每行向量与matrix2中所有行向量的余弦相似度
    return torch.from_numpy(cosine_similarity(matrix1.numpy(),
                                              matrix2.numpy()))


def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return "    ".join([
        str(metric) + "=" + f"{value:.4f}"
        for metric, value in result_dict.items()
    ])


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur
