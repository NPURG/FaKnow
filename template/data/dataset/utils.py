import os

import torch.utils.data
from torch.utils.data import ConcatDataset, random_split
from torchvision.datasets.folder import find_classes, is_image_file
from typing import Optional, Callable, Dict, List, Tuple, Union, Set


def default_walker(class_dir: str, class_name: str, class_index: int,
                   instances: List[Tuple], available_classes: Set):
    for root, _, file_names in sorted(os.walk(class_dir, followlinks=True)):
        for file_name in sorted(file_names):
            path = os.path.join(root, file_name)
            instances.append((path, class_index))

            if class_name not in available_classes:
                available_classes.add(class_name)


def walker_with_images(class_dir: str, class_name: str, class_index: int,
                       instances: List[Tuple], available_classes: Set):
    for root, _, file_names in sorted(os.walk(class_dir, followlinks=True)):
        find_text = find_image = False
        for file_name in sorted(file_names):
            if is_image_file(file_name):
                image_path = os.path.join(root, file_name)
                find_image = True
            else:
                text_path = os.path.join(root, file_name)
                find_text = True

            if find_text and find_image:
                instances.append((text_path, image_path, class_index))
                find_text = find_image = False

            if class_name not in available_classes:
                available_classes.add(class_name)


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    walk_class_dir: Optional[Callable[[str, str, int, List[Tuple], Set],
                                      None]] = default_walker
) -> List[Tuple[str, int]]:
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError(
            "'class_to_index' must have at least one entry to collect any samples."
        )

    # both_none = extensions is None and is_valid_file is None
    # both_something = extensions is not None and is_valid_file is not None
    # if both_none or both_something:
    #     raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    #
    # if extensions is not None:
    #     def is_valid_file(x: str) -> bool:
    #         return has_file_allowed_extension(x, extensions)
    #
    # is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for class_name in sorted(class_to_idx.keys()):
        class_index = class_to_idx[class_name]
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue

        walk_class_dir(class_dir, class_name, class_index, instances,
                       available_classes)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


def re_split_dataset(datasets: List[torch.utils.data.Dataset],
                     lengths: List[int]):
    all_dataset = ConcatDataset(datasets)
    return random_split(all_dataset, lengths)
