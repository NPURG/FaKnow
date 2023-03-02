from typing import Optional, Callable, Tuple, List

import torch
from torch import Tensor, nn

from model.layers.layer import TextCNNLayer
from template.model.model import AbstractModel

"""
SAFE: Similarity-Aware Multi-Modal Fake News Detection
paper: https://arxiv.org/pdf/2003.04981v1.pdf
code: https://github.com/Jindi0/SAFE
"""


def cos_loss_func(cos_dis_sim: Tensor, label: Tensor) -> float:
    return -(label * cos_dis_sim.log()).sum(1).mean()


class _TextCNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 filter_num: int,
                 kernel_sizes: List[int],
                 dropout: float,
                 output_size: int):
        super().__init__()
        self.text_ccn_layer = TextCNNLayer(input_size, filter_num, kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filter_num, output_size)

    def forward(self, x: torch.Tensor):
        x = self.text_ccn_layer(x)
        x = self.dropout(x)
        return self.fc(x)


class SAFE(AbstractModel):
    def __init__(
            self,
            embedding_size: int = 300,
            conv_in_size: int = 32,
            filter_num: int = 128,
            conv_out_size: int = 200,
            dropout: float = 0.,
            loss_funcs: Optional[List[Callable]] = None,
            loss_weights: Optional[List[float]] = None,
    ):
        super(SAFE, self).__init__()

        if loss_funcs is None:
            loss_funcs = [nn.CrossEntropyLoss(), cos_loss_func]
        self.loss_funcs = loss_funcs
        if loss_weights is None:
            loss_weights = [1.0, 1.0]
        self.loss_weights = loss_weights

        self.embedding_size = embedding_size

        self.reduce = nn.Linear(embedding_size, conv_in_size)
        nn.init.trunc_normal_(self.reduce.weight, std=0.1)
        nn.init.constant_(self.reduce.bias, 0.1)

        filter_sizes = [3, 4]
        self.head_block = _TextCNN(conv_in_size, filter_num, filter_sizes, dropout, conv_out_size)
        self.body_block = _TextCNN(conv_in_size, filter_num, filter_sizes, dropout, conv_out_size)
        self.image_block = _TextCNN(conv_in_size, filter_num, filter_sizes, dropout, conv_out_size)

        self.predictor = nn.Linear(conv_out_size * 3, 2)
        nn.init.trunc_normal_(self.predictor.weight, std=0.1)
        nn.init.constant_(self.predictor.bias, 0.1)

    def forward(
            self,
            head: torch.Tensor,
            body: torch.Tensor,
            image: torch.Tensor,
    ):

        head = self.reduce(head)
        body = self.reduce(body)
        image = self.reduce(image)

        headline_vectors = self.head_block(head)
        body_vectors = self.body_block(body)
        image_vectors = self.image_block(image)

        combine_images = torch.cat([image_vectors, image_vectors], dim=1)
        combine_texts = torch.cat([headline_vectors, body_vectors], dim=1)

        combine_images_norm = combine_images.norm(p=2, dim=1)
        combine_texts_norm = combine_texts.norm(p=2, dim=1)

        image_text = (combine_images * combine_texts).sum(1)

        # cos similarity
        cosine_similarity = (
                                    1 + (image_text / (combine_images_norm * combine_texts_norm + 1e-8))
                            ) / 2
        cosine_distance = 1 - cosine_similarity
        cos_dis_sim = torch.stack([cosine_distance, cosine_similarity], 1)

        cat_vectors = torch.cat([headline_vectors, body_vectors, image_vectors], dim=1)
        class_output = self.predictor(cat_vectors)

        return class_output, cos_dis_sim

    def calculate_loss(self, data: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[torch.Tensor, str]:
        headline, body, image, label = data
        class_output, cos_dis_sim = self.forward(headline, body, image)

        class_loss = self.loss_funcs[0](class_output, label.long()) * self.loss_weights[0]

        label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=2)
        cos_dis_sim_loss = self.loss_funcs[1](cos_dis_sim.to(torch.float32), label.to(torch.float32)) * \
                           self.loss_weights[1]

        loss = class_loss + cos_dis_sim_loss
        msg = f"class_loss={class_loss}, cos_dis_sim_loss={cos_dis_sim_loss}"

        return loss, msg

    def predict(self, data_without_label: Tuple[Tensor, Tensor, Tensor]) -> torch.Tensor:
        head, body, image = data_without_label
        class_output, _ = self.forward(head, body, image)
        return torch.softmax(class_output, dim=-1)
