import os
import torch
import torch.utils.data
from torchvision.datasets.folder import find_classes
from typing import Optional, Callable, Dict, List, Tuple, Any, Union, Set
from template.data.dataset.utils import make_dataset, default_walker


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, root: str,
                 word_embedding: Optional[Callable] = None,
                 other_params: Optional[Dict[str, Any]] = None,
                 walk_class_dir: Optional[Callable[[str, str, int, List[Tuple], Set], None]] = default_walker):
        self.classes, self.class_to_idx = find_classes(root)
        # self.samples = self.make_dataset(root, self.class_to_idx, None, None)
        self.samples = make_dataset(root, self.class_to_idx, None, None, walk_class_dir)

        self.texts = [sample[0] for sample in self.samples]
        self.labels = [sample[1] for sample in self.samples]
        self.word_embedding = word_embedding
        self.other_params = other_params

    def __getitem__(self, index) -> Tuple[Any, Any]:
        text = self.samples[index][0]
        label = self.samples[index][-1]
        if self.word_embedding is not None:
            text = self.word_embedding(text)

        return text, label

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Optional[Dict[str, int]] = None,
            extensions: Optional[Union[str, Tuple[str, ...]]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            walk_class_dir: Optional[Callable[[str, str, int, List[Tuple], Set], None]] = default_walker
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

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

            walk_class_dir(class_dir, class_name, class_index, instances, available_classes)
            # for root, _, file_names in sorted(os.walk(class_dir, followlinks=True)):
            #     for file_name in sorted(file_names):
            #         path = os.path.join(root, file_name)
            #         instances.append((path, class_index))
            #
            #         if class_name not in available_classes:
            #             available_classes.add(class_name)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances
