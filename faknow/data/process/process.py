import random
from typing import List, Callable, Any, Tuple
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset, random_split
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ViTImageProcessor

from faknow.data.dataset.text import TextDataset
from faknow.data.dataset.multi_modal import MultiModalDataset


def split_dataset(
        data_path: str,
        text_features: List[str],
        tokenize: Callable[[List[str]], Any],
        image_features: List[str] = None,
        transform: Callable[[str], Any] = None,
        ratio: List[float] = None,
) -> List[Subset[Any]]:
    """
    split TextDataset or MultiModalDataset with given ratio.
    If image_features is None, split TextDataset, else split MultiModalDataset.

    Args:
        data_path (str): path to json file
        text_features (List[str]): a list of names of text features in files
        tokenize (Callable[[List[str]], Any]): function to tokenize text,
            which takes a list of texts
            and returns a tensor or a dict of tensors
        image_features (List[str]): a list of names of image features in files.
            Default=None.
        transform (Callable[[str], Any]): function to transform image,
            which takes a path to image
            and returns a tensor or a dict of tensors.
            Default=None.
        ratio (List[float]): a list of ratios of subset.
            If None, default to [0.7, 0.1, 0.2]. Default=None.

    Returns:
        subsets (List[Subset[Any]]): a list of subsets of dataset
    """

    if ratio is None:
        ratio = [0.7, 0.1, 0.2]
    else:
        error_msg = "ratio must be a list of positive numbers whose sum is 1"
        for i in ratio:
            assert i > 0, error_msg
        assert sum(ratio) == 1, error_msg

    if image_features is None:
        dataset = TextDataset(data_path, text_features, tokenize)
    else:
        dataset = MultiModalDataset(
            data_path, text_features, tokenize, image_features, transform
        )

    sizes = [int(len(dataset) * i) for i in ratio[:-1]]
    sizes.append(len(dataset) - sum(sizes))

    return random_split(dataset, sizes)


def lsh_data_selection(domain_embeddings: torch.Tensor,
                       labelling_budget=100,
                       hash_dimension=10) -> List[int]:
    """
    Local sensitive hash (LSH) selection for training dataset.

    Args:
        domain_embeddings (torch.Tensor): 2-D domain embedding tensor of samples.
        labelling_budget (int): Number of selection budget, must be smaller than the number of samples. Default=100.
        hash_dimension (int): Dimension of random hash vector. Default=10.

    Returns:
        List[int]: A list of selected sample indices.
    """
    if len(domain_embeddings.shape) != 2:
        raise TypeError("domain embedding must be 2-D tensor!")

    if labelling_budget > domain_embeddings.shape[0]:
        raise RuntimeError(
            f"labelling budget({labelling_budget}) is greater than data pool size({domain_embeddings.shape[0]})"
        )

    embedding_size = domain_embeddings.shape[1]
    final_selected_ids = []
    is_final_selected = defaultdict(lambda: False)

    random_distribution = [3 ** 0.5, 0.0, 0.0, 0.0, 0.0, -(3 ** 0.5)]

    while len(final_selected_ids) < labelling_budget:
        # Generate random vectors
        random_vectors = []
        for hash_run in range(hash_dimension):
            vec = random.choices(random_distribution, k=embedding_size)
            random_vectors.append(torch.tensor(vec))

        # Create hash table
        code_dict = defaultdict(lambda: [])  # {str(h-dim hash value): [domain_id]}
        for i, item in enumerate(domain_embeddings):
            code = ""
            # Skip if the item is already selected
            if is_final_selected[i]:
                continue

            # Concat result of hash functions to generate hash vectors
            for code_vec in random_vectors:
                code = code + str(int(torch.dot(item, code_vec) > 0))
            code_dict[code].append(i)

        selected_ids = []
        is_selected = defaultdict(lambda: False)

        # Pick one item from each item bin
        for item in code_dict:
            selected_item = random.choice(code_dict[item])
            selected_ids.append(selected_item)  # 添加domain id，即domain embedding中的行号
            is_selected[selected_item] = True

        # Remove a set of instances randomly to meet the labelling budget
        if len(final_selected_ids + selected_ids) > labelling_budget:
            random_pick_size = labelling_budget - len(final_selected_ids)
            mod_selected_ids = []
            for z in range(random_pick_size):
                select_item = random.choice(selected_ids)
                mod_selected_ids.append(select_item)
                selected_ids.remove(select_item)
            final_selected_ids = final_selected_ids + mod_selected_ids
            return final_selected_ids

        for item in selected_ids:
            is_final_selected[item] = True
        final_selected_ids = final_selected_ids + selected_ids
    return final_selected_ids


class ProcessorForVit:
    """
    processor for ViT
    """

    def __init__(self, vit_name: str, image_mean: List[float],
                 image_std: List[float]):
        """
        Args:
            vit_name (str): name of ViT model
            image_mean (List[float]): mean of image
            image_std (List[float]): std of image
        """
        self.processor = ViTImageProcessor.from_pretrained(
            vit_name, image_mean=image_mean, image_std=image_std)

    def __call__(self, path: str) -> torch.Tensor:
        """
        transform image to tensor for ViT

        Args:
            img (Image.Image): image

        Returns:
            torch.Tensor: tensor of the image, shape=(3, 224, 224)
        """
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
            return self.processor(img,
                                  return_tensors="pt").pixel_values.squeeze(0)


def calculate_cos_matrix(matrix1: torch.Tensor, matrix2: torch.Tensor):
    """
    Calculate the pairwise cosine similarity matrix between two matrices.

    Args:
        matrix1 (torch.Tensor): The first matrix, shape=(n, d)
        matrix2 (torch.Tensor): The second matrix, shape=(m, d)

    Returns:
        torch.Tensor: The cosine similarity matrix, shape=(n, m)
    """
    return torch.from_numpy(
        cosine_similarity(matrix1.cpu().numpy(), matrix2.cpu().numpy())
    ).to(device=matrix1.device)


class DropEdge:
    """
    randomly drop out edges for BiGCN
    """

    def __init__(self, td_drop_rate: float, bu_drop_rate: float):
        """
        Args:
            td_drop_rate (float): drop rate of drop edge in top-down direction
            bu_drop_rate (float): drop rate of drop edge in bottom-up direction
        """

        self.td_drop_rate = td_drop_rate
        self.bu_drop_rate = bu_drop_rate

    @staticmethod
    def _random_sample(
            row: List[int], col: List[int], drop_rate: float
    ) -> Tuple[List[int], List[int]]:
        """

        Args:
            row (List[int]): a list of origins in edges index
            col (List[int]): a list of destinations in edges index
            drop_rate (float): drop out rate

        Returns:
            Tuple[List[int], List[int]]: edges index after drop out
        """

        length = len(row)
        pos_list = random.sample(range(length), int(length * (1 - drop_rate)))
        pos_list = sorted(pos_list)
        new_row = list(np.array(row)[pos_list])
        new_col = list(np.array(col)[pos_list])

        return new_row, new_col

    def __call__(self, data: Data) -> Data:
        """
        Args:
            data (Data): The data in pyg.

        Returns:
            Batch: The data with dropped edges.
        """

        edge_index = data.edge_index

        if self.td_drop_rate > 0:
            row, col = self._random_sample(
                list(edge_index[0]), list(edge_index[1]), self.td_drop_rate
            )
            new_edge_index = [row, col]
        else:
            new_edge_index = edge_index

        bu_row = list(edge_index[1])
        bu_col = list(edge_index[0])
        if self.bu_drop_rate > 0:
            row, col = self._random_sample(bu_row, bu_col, self.bu_drop_rate)
            bu_new_edge_index = [row, col]
        else:
            bu_new_edge_index = [bu_row, bu_col]

        data.edge_index = torch.LongTensor(new_edge_index)
        data.BU_edge_index = torch.LongTensor(bu_new_edge_index)
        data.root = torch.FloatTensor(data.x[0])
        data.root_index = torch.LongTensor([0])

        return data
