from typing import List, Callable, Any

from torch.utils.data import Subset, random_split

from faknow.data.dataset.text import TextDataset
from faknow.data.dataset.multi_modal import MultiModalDataset


def split_dataset(data_path: str,
                  text_features: List[str],
                  tokenize: Callable[[List[str]], Any],
                  image_features: List[str] = None,
                  transform: Callable[[str], Any] = None,
                  ratio: List[float] = None) -> List[Subset[Any]]:
    """
    split TextDataset or MultiModalDataset with given ratio.
    If image_features is None, split TextDataset, else split MultiModalDataset.

    Args:
        data_path (str): path to json file
        text_features (List[str]): a list of names of text features in json file
        tokenize (Callable[[List[str]], Any]): function to tokenize text,
            which takes a list of texts and returns a tensor or a dict of tensors
        image_features (List[str]): a list of names of image features in json file.
            Default=None.
        transform (Callable[[str], Any]): function to transform image,
            which takes a path to image and returns a tensor or a dict of tensors.
            Default=None.
        ratio (List[float]): a list of ratios of subset.
            If None, default to [0.7, 0.1, 0.2]. Default=None.

    Returns:
        subsets (List[Subset[Any]]): a list of subsets of dataset
    """

    if ratio is None:
        ratio = [0.7, 0.1, 0.2]
    else:
        error_msg = 'ratio must be a list of positive numbers whose sum is 1'
        for i in ratio:
            assert i > 0, error_msg
        assert sum(ratio) == 1, error_msg

    if image_features is None:
        dataset = TextDataset(data_path, text_features, tokenize)
    else:
        dataset = MultiModalDataset(data_path, text_features, tokenize, image_features, transform)

    sizes = [int(len(dataset) * i) for i in ratio[:-1]]
    sizes.append(len(dataset) - sum(sizes))

    return random_split(dataset, sizes)