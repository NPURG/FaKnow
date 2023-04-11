from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from faknow.model.layers.layer import TextCNNLayer
from faknow.model.model import AbstractModel

"""
SAFE: Similarity-Aware Multi-Modal Fake News Detection
paper: https://arxiv.org/pdf/2003.04981v1.pdf
code: https://github.com/Jindi0/SAFE
"""


def loss_func(cos_dis_sim: Tensor, label: Tensor) -> Tensor:
    return -(F.one_hot(label.long(), num_classes=2).float() * cos_dis_sim.log()).sum(1).mean()


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
    """
    SAFE: Similarity-Aware Multi-Modal Fake News Detection
    """

    def __init__(
            self,
            embedding_size: int = 300,
            conv_in_size: int = 32,
            filter_num: int = 128,
            conv_out_size: int = 200,
            dropout: float = 0.,
            loss_weights: Optional[List[float]] = None,
    ):
        """

        Args:
            embedding_size (int): embedding size of text.
            conv_in_size (int): number of in channels in TextCNNLayer. Default=32
            filter_num (int): number of filters in TextCNNLayer. Default=128
            conv_out_size (int): number of out channels in TextCNNLayer. Default=200
            dropout (float): drop out rate. Default=0.0
            loss_weights (List[float]): list of loss weights. Default=[1.0, 1.0]
        """
        super(SAFE, self).__init__()

        self.loss_funcs = [loss_func, loss_func]
        if loss_weights is None:
            loss_weights = [1.0, 1.0]
        self.loss_weights = loss_weights

        self.embedding_size = embedding_size

        self.reduce = nn.Linear(embedding_size, conv_in_size)

        filter_sizes = [3, 4]
        self.head_block = _TextCNN(conv_in_size, filter_num, filter_sizes, dropout, conv_out_size)
        self.body_block = _TextCNN(conv_in_size, filter_num, filter_sizes, dropout, conv_out_size)
        self.image_block = _TextCNN(conv_in_size, filter_num, filter_sizes, dropout, conv_out_size)

        self.predictor = nn.Linear(conv_out_size * 3, 2)

        self.__init_weights__()

    def __init_weights__(self):
        nn.init.trunc_normal_(self.reduce.weight, std=0.1)
        nn.init.constant_(self.reduce.bias, 0.1)
        nn.init.trunc_normal_(self.predictor.weight, std=0.1)
        nn.init.constant_(self.predictor.bias, 0.1)

    def forward(
            self,
            head: torch.Tensor,
            body: torch.Tensor,
            image: torch.Tensor,
    ):
        """

        Args:
            head (Tensor): embedded title of post, shape=(batch_size, title_len, embedding_size)
            body (Tensor): embedded content of post, shape=(batch_size, content_len, embedding_size)
            image (Tensor): embedded sentence converted from post image, shape=(batch_size, sentence_len, embedding_size)

        Returns:
            tuple:
                - class_output (Tensor): prediction of being fake news, shape=(batch_size, 2)
                - cos_dis_sim (Tensor): prediction of belonging to which domain, shape=(batch_size, 2)
        """
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
        class_output = torch.softmax(class_output, dim=-1)

        return class_output, cos_dis_sim

    def calculate_loss(self, data) -> Tuple[torch.Tensor, Dict[str, float]]:
        headline = data['head']
        body = data['body']
        image = data['image']
        label = data['label']
        class_output, cos_dis_sim = self.forward(headline, body, image)

        class_loss = self.loss_funcs[0](class_output, label) * self.loss_weights[0]
        cos_dis_sim_loss = self.loss_funcs[1](cos_dis_sim, label) * self.loss_weights[1]

        loss = class_loss + cos_dis_sim_loss
        return loss, {'class_loss': class_loss.item(), 'cos_dis_sim_loss': cos_dis_sim_loss.item()}

    def predict(self, data_without_label) -> torch.Tensor:
        head = data_without_label['head']
        body = data_without_label['body']
        image = data_without_label['image']
        class_output, _ = self.forward(head, body, image)
        return class_output
