from typing import Optional, Callable, Tuple, List

import torch
from torch import Tensor, nn

from template.model.model import AbstractModel

"""
SAFE: Similarity-Aware Multi-Modal Fake News Detection
paper: https://arxiv.org/pdf/2003.04981v1.pdf
code: https://github.com/Jindi0/SAFE
"""


def cos_loss_func(cos_dis_sim: Tensor, label: Tensor) -> float:
    return -(label * cos_dis_sim.log()).sum(1).mean()


# convolution layers
class _ConvBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            input_len: int,
            num_filters: int,
            dropout_prob: float,
            final_len: int,
    ):
        super().__init__()

        # filter size: 3
        self.conv3 = nn.Conv2d(1, num_filters, kernel_size=(3, input_len))
        self.pool3 = nn.MaxPool2d(
            kernel_size=(input_size - 3 + 1, 1),
            stride=(1, 1),
        )

        # filter size: 4
        self.conv4 = nn.Conv2d(1, num_filters, kernel_size=(4, input_len))
        self.pool4 = nn.MaxPool2d(
            kernel_size=(input_size - 4 + 1, 1),
            stride=(1, 1),
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(num_filters * 2, final_len)

    def forward(self, x: torch.Tensor):
        # bs, head_size, input_len -> bs, 1, head_size, input_len
        x = x.unsqueeze(1)

        # filter size: 3
        x3 = self.conv3(x)
        x3 = self.pool3(x3)

        # filter size: 3
        x4 = self.conv4(x)
        x4 = self.pool4(x4)

        x = torch.cat([x3, x4], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, x.shape[-1])

        x = self.dropout(x)

        return self.fc(x)


class SAFE(AbstractModel):
    def __init__(
            self,
            head_size: int = 30,
            body_size: int = 100,
            image_size: int = 66,
            embedding_size: int = 300,
            input_len: int = 32,
            num_filters: int = 128,
            final_len: int = 200,
            dropout_prob: float = 0.,
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
        self.input_len = input_len
        self.dropout_prob = dropout_prob

        self.reduce = nn.Linear(embedding_size, input_len)
        nn.init.trunc_normal_(self.reduce.weight, std=0.1)
        nn.init.constant_(self.reduce.bias, 0.1)

        self.head_block = _ConvBlock(
            input_size=head_size,
            input_len=input_len,
            num_filters=num_filters,
            dropout_prob=dropout_prob,
            final_len=final_len,
        )

        self.body_block = _ConvBlock(
            input_size=body_size,
            input_len=input_len,
            num_filters=num_filters,
            dropout_prob=dropout_prob,
            final_len=final_len,
        )

        self.image_block = _ConvBlock(
            input_size=image_size,
            input_len=input_len,
            num_filters=num_filters,
            dropout_prob=dropout_prob,
            final_len=final_len,
        )

        # todo 对于类别输出再加一个softmax
        self.predictor = nn.Linear(final_len * 3, 2)
        nn.init.trunc_normal_(self.predictor.weight, std=0.1)
        nn.init.constant_(self.predictor.bias, 0.1)

    def forward(
            self,
            x_heads: torch.Tensor,
            x_bodies: torch.Tensor,
            x_images: torch.Tensor,
    ):

        x_heads = self.reduce(x_heads)
        x_bodies = self.reduce(x_bodies)
        x_images = self.reduce(x_images)

        headline_vectors = self.head_block(x_heads)
        body_vectors = self.body_block(x_bodies)
        image_vectors = self.image_block(x_images)

        # cos similarity
        combine_images = torch.cat([image_vectors, image_vectors], dim=1)
        combine_texts = torch.cat([headline_vectors, body_vectors], dim=1)

        combine_images_norm = combine_images.norm(p=2, dim=1)
        combine_texts_norm = combine_texts.norm(p=2, dim=1)

        image_text = (combine_images * combine_texts).sum(1)

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

        # pytorch内置的交叉熵，需要label为n*1，而不是n*2
        class_loss = self.loss_funcs[0](class_output, label.long()) * self.loss_weights[0]
        # todo cos_dis_sim_loss 即 loss2，注意维度对齐
        label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=2)
        cos_dis_sim_loss = self.loss_funcs[1](cos_dis_sim.to(torch.float32), label.to(torch.float32)) * self.loss_weights[1]

        loss = class_loss + cos_dis_sim_loss

        msg = f"class_loss={class_loss}, cos_dis_sim_loss={cos_dis_sim_loss}"

        return loss, msg

    def predict(self, data_without_label: Tuple[Tensor, Tensor, Tensor]) -> torch.Tensor:
        x_heads, x_bodies, x_images = data_without_label
        class_output, _ = self.forward(x_heads, x_bodies, x_images)
        return class_output
