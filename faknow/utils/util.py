import datetime
import random
import warnings
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Batch


def calculate_cos_matrix(matrix1: torch.Tensor, matrix2: torch.Tensor):
    """
    Calculate the cosine similarity matrix between two matrices.

    Args:
        matrix1 (torch.Tensor): The first matrix.
        matrix2 (torch.Tensor): The second matrix.

    Returns:
        torch.Tensor: The cosine similarity matrix.
    """
    return torch.from_numpy(cosine_similarity(matrix1.numpy(),
                                              matrix2.numpy()))


def dict2str(result_dict: Dict[str, float]) -> str:
    """
    Convert a dictionary of metrics to a string.

    Args:
        result_dict (dict): A dictionary containing metric names and corresponding values.

    Returns:
        str: The formatted string representation of the dictionary.
    """

    return "    ".join([
        str(metric) + "=" + f"{value:.6f}"
        for metric, value in result_dict.items()
    ])


def now2str() -> str:
    """
    Get the current time and convert it to a formatted string.

    Returns:
        str: The current time in the format '%Y-%m-%d-%H_%M_%S'.
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%Y-%m-%d-%H_%M_%S")

    return cur


def seconds2str(seconds: float) -> str:
    """
    Convert seconds to a human-readable time format.

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: The duration in the format 'h:mm:ss' or 'm:ss'.
    """
    if seconds < 60:
        return f'{seconds:.6f}s'

    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if m == 0:
        return f'{s}s'
    elif h == 0:
        return f'{m}m{s}s'
    return f'{h}h{m}m{s}s'


def check_loss_type(result):
    """
    Check the type of the loss and convert it to a tensor if necessary.

    Args:
        result (Union[torch.Tensor, dict]): The loss value or a dictionary of losses.

    Returns:
        Tuple[torch.Tensor, bool]: A tuple containing the loss tensor and a boolean indicating if the loss was a dictionary.
    """
    result_is_dict = False

    if type(result) is dict:
        result_is_dict = True
        if 'total_loss' in result.keys():
            loss = result['total_loss']
        else:
            # todo 是否允许没有total_loss，采用所有loss的和作为total_loss
            warnings.warn(
                f"no total_loss in result: {result}, use sum of all losses as total_loss"
            )
            loss = torch.sum(torch.stack(list(result.values())))
    elif type(result) is torch.Tensor:
        loss = result
    else:
        raise TypeError(f"result type error: {result}")

    return loss, result_is_dict


def lsh_data_selection(domain_embeddings: torch.Tensor, labelling_budget=100, hash_dimension=10) -> List[int]:
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
            f"labelling budget({labelling_budget}) is greater than data pool size({domain_embeddings.shape[0]})")

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
            code = ''
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


def read_stop_words(path: str) -> List[str]:
    """
    Read stop words from a file.

    Args:
        path (str): The path to the file containing stop words.

    Returns:
        List[str]: A list of stop words.
    """
    with open(path, 'r', encoding='utf-8') as f:
        stop_words = [str(line).strip() for line in f.readlines()]
    return stop_words


class EarlyStopping:
    """
    Early stopping to stop the training when the score does not improve after
    certain epochs.
    """

    def __init__(self, patience=10, delta=0.000001, mode='max'):
        """
        Args:
            patience (int): number of epochs to wait for improvement, default=10
            delta (float): minimum change in the monitored quantity to qualify as an improvement, default=0.000001
            mode (str): minimize or maximize score, one of {min, max}, default=max
        """

        assert mode in ['min', 'max'], "mode must be either 'min' or 'max'"

        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, current_score: float) -> bool:
        """
        Check if the current score is the best score and update early stopping status.

        Args:
            current_score (float): The current score to check if it is the best score.

        Returns:
            bool: Whether the current score is the best score.
        """
        improvement = False
        if self.best_score is None:
            improvement = True
        elif self.mode == 'min':
            if current_score < self.best_score - self.delta:
                improvement = True
        elif self.mode == 'max':
            if current_score > self.best_score + self.delta:
                improvement = True

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return improvement


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

    def __call__(self, data: Batch) -> Batch:
        """
        Args:
            data (Batch): The batch data in pyg.

        Returns:
            Batch: The batch data with dropped edges.
        """

        edge_index = data.edge_index

        if self.td_drop_rate > 0:
            row = list(edge_index[0])
            col = list(edge_index[1])
            length = len(row)
            poslist = random.sample(range(length),
                                    int(length * (1 - self.td_drop_rate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edge_index = [row, col]
        else:
            new_edge_index = edge_index

        bu_row = list(edge_index[1])
        bu_col = list(edge_index[0])
        if self.bu_drop_rate > 0:
            length = len(bu_row)
            poslist = random.sample(range(length),
                                    int(length * (1 - self.bu_drop_rate)))
            poslist = sorted(poslist)
            row = list(np.array(bu_row)[poslist])
            col = list(np.array(bu_col)[poslist])
            bu_new_edge_index = [row, col]
        else:
            bu_new_edge_index = [bu_row, bu_col]

        data.edge_index = torch.LongTensor(new_edge_index)
        data.BU_edge_index = torch.LongTensor(bu_new_edge_index)
        data.root = torch.FloatTensor(data.x[0])
        data.root_index = torch.LongTensor([0])

        return data
