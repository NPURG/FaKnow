import math
import copy
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from faknow.model.model import AbstractModel


class CAFE(AbstractModel):
    r"""
    CAFE: Cross-modal Ambiguity Learning for Multimodal Fake News Detection, WWW 2022
    paper: https://dl.acm.org/doi/10.1145/3485447.3511968
    code: https://github.com/cyxanna/CAFE
    """
    def __init__(self, feature_dim=64 + 16 + 16, h_dim=64):
        """
        Args:
            feature_dim (int): number of feature dim
            h_dim (int): number of hidden dim
        """
        super(CAFE, self).__init__()
        self.encoding = _TextImageEncoder()
        self.ambiguity_module = _AmbiguityModule()
        self.uni_modal = _UnimodalModule()
        self.cross_modal = _CrossModalModule()
        self.loss_func_detection = torch.nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(nn.Linear(feature_dim, h_dim),
                                        nn.BatchNorm1d(h_dim), nn.ReLU(),
                                        nn.Linear(h_dim, h_dim),
                                        nn.BatchNorm1d(h_dim), nn.ReLU(),
                                        nn.Linear(h_dim, 2))
        self.similarity_module = _SimilarityModule()

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> Tensor:
        """
        Args:
            text (Tensor): the raw text, shape=(batch_size, 30, 200)
            image (Tensor): the raw image, shape=(batch_size, 512)
        Returns:
            Tensor: prediction of being fake news, shape=(batch_size, 2)
        """
        text_aligned, image_aligned, _ = self.similarity_module(text, image)
        skl = self.ambiguity_module(text_aligned, image_aligned)
        correlation = self.cross_modal(text_aligned, image_aligned)
        text_prime, image_prime = self.encoding(text, image)
        text_prime, image_prime = self.uni_modal(text_prime, image_prime)

        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        text_final = weight_uni * text_prime
        img_final = weight_uni * image_prime
        corre_final = weight_corre * correlation
        final_corre = torch.cat([text_final, img_final, corre_final], 1)
        out = self.classifier(final_corre)
        return out

    def calculate_loss(self, data: Dict[str, Tensor]) -> Tensor:
        """
        process raw data using similarity_module
        calculate loss via CrossEntropyLoss
        Args:
            data (Tuple[Tensor]): batch data tuple,including text, image and label
        Returns:
            torch.Tensor: loss
        """
        text = data['text']
        image = data['image']
        label = data['label']
        pre_detection = self.forward(text, image)
        loss_detection = self.loss_func_detection(pre_detection, label)
        return loss_detection

    def predict(self, data: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            data (Tuple[Tensor]): batch data tuple, including text, image and label
        Returns:
            Tensor: softmax probability, shape=(batch_size, 2)
        """
        text = data['text']
        image = data['image']
        out = self.forward(text, image)
        return torch.softmax(out, dim=-1)


class _FastTextCNN(nn.Module):
    # a CNN-based altertative approach of bert for text encoding
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        """
        Args:
            channel (int): the number of conv channel, default=32
            kernel_size:  (int): the size of cnn kernel, default=(1, 2, 4, 8)
        """
        super(_FastTextCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(nn.Conv1d(200, channel, kernel_size=kernel),
                              nn.BatchNorm1d(channel), nn.ReLU(),
                              nn.AdaptiveMaxPool1d(1)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): processed text data, shape=(batch_size,30,200)
        Returns:
            Tensor: FastCNN data,shape=(batch_size,128)
        """
        x = x.permute(0, 2, 1)
        x_out = []
        for module in self.fast_cnn:
            if (x.shape[0] == 1):
                x_out.append(module(x).view(1, -1))
            else:
                x_out.append(module(x).squeeze())
        x_out = torch.cat(x_out, 1)
        return x_out


class _TextImageEncoder(nn.Module):
    """
    encoding for input text and image
    """
    def __init__(self,
                 cnn_channel=32,
                 cnn_kernel_size=(1, 2, 4, 8),
                 shared_image_dim=128,
                 shared_text_dim=128):
        """
        Args:
            cnn_channel (int): the number of cnn channel, default=32
            cnn_kernel_size (int): the size of cnn kernel, default=(1, 2, 4, 8)
            shared_image_dim: output dim of image data
            shared_text_dim: output dim of text data
        """
        super(_TextImageEncoder, self).__init__()
        self.shared_text_encoding = _FastTextCNN(channel=cnn_channel,
                                                 kernel_size=cnn_kernel_size)
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(),
            nn.Linear(64, shared_text_dim), nn.BatchNorm1d(shared_text_dim),
            nn.ReLU())
        self.shared_image = nn.Sequential(nn.Linear(512, 256),
                                          nn.BatchNorm1d(256), nn.ReLU(),
                                          nn.Dropout(),
                                          nn.Linear(256, shared_image_dim),
                                          nn.BatchNorm1d(shared_image_dim),
                                          nn.ReLU())

    def forward(self, text: Tensor, image: Tensor):
        """
        Args:
            text (Tensor): batch text data, shape=(batch_size, 30, 200)
            image (Tensor): batch image data, shape=(batch_size, 512)
        Returns:
            text_shared (Tensor): Encoding text data, shape=(batch_size, 128)
            image_shared (Tensor): Encoding image data, shape=(batch_size, 128)
        """
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class _SimilarityModule(AbstractModel):
    """
    Cross modal aligment and calculate cosine embedding loss
    """
    def __init__(self, shared_dim=128, sim_dim=64):
        """
        Args:
            shared_dim (int): dimension of aligner node feature
            sim_dim (int): dimension of similarity node feature
        """
        super(_SimilarityModule, self).__init__()
        self.encoding = _TextImageEncoder()
        self.loss_func_similarity = torch.nn.CosineEmbeddingLoss()
        self.text_aligner = nn.Sequential(nn.Linear(shared_dim, shared_dim),
                                          nn.BatchNorm1d(shared_dim),
                                          nn.ReLU(),
                                          nn.Linear(shared_dim, sim_dim),
                                          nn.BatchNorm1d(sim_dim), nn.ReLU())
        self.image_aligner = nn.Sequential(nn.Linear(shared_dim, shared_dim),
                                           nn.BatchNorm1d(shared_dim),
                                           nn.ReLU(),
                                           nn.Linear(shared_dim, sim_dim),
                                           nn.BatchNorm1d(sim_dim), nn.ReLU())
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Linear(64, 2))

    def forward(self, text: Tensor, image: Tensor):
        """
        Args:
            text (Tensor): text data, shape=(batch_size, 30, 200)
            image (Tensor): image data, shape=(batch_size, 512)
        Returns:
            text_aligned (Tensor): aligned text, shape=(batch_size, 64)
            image_aligned (Tensor): aligned image, shape=(batch_size, 64)
            pred_similarity (Tensor): probability, shape=(batch_size, 2)
        """
        text_encoding, image_encoding = self.encoding(text, image)
        text_aligned = self.text_aligner(text_encoding)
        image_aligned = self.image_aligner(image_encoding)
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, image_aligned, pred_similarity

    def calculate_loss(self, data: Tuple[Tensor]) -> Tensor:
        """
        calculate loss via CosineEmbeddingLoss
        Args:
            data Tuple[Tensor]: batch data, including text, image, label
        Returns:
            torch.Tensor: CosineEmbeddingLoss
        """
        fixed_text, matched_image, unmatched_image = self.sample_data_pairs(
            data)

        text_aligned_match, image_aligned_match, similarity_match = self.forward(
            fixed_text, matched_image)
        text_aligned_unmatch, image_aligned_unmatch, similarity_unmatch = self.forward(
            fixed_text, unmatched_image)

        similarity_label = torch.cat([
            torch.ones(similarity_match.shape[0],
                       device=similarity_match.device),
            -1 * torch.ones(similarity_unmatch.shape[0],
                            device=similarity_unmatch.device)
        ],
                                     dim=0)
        text_aligned = torch.cat([text_aligned_match, text_aligned_unmatch],
                                 dim=0)
        image_aligned = torch.cat([image_aligned_match, image_aligned_unmatch],
                                  dim=0)
        loss_similarity = self.loss_func_similarity(text_aligned,
                                                    image_aligned,
                                                    similarity_label)
        return loss_similarity

    def sample_data_pairs(self, data: Dict[str, Tensor]) -> Tuple[Tensor]:
        """
        randomly sample positive text-image pairs and negative text-image pairs

        Args:
            data (Tuple[Tensor, any]): batch data, including text, image, label
        Returns:
            fixed_text (Tensor): processed text data, shape=(sample_num, 30, 200)
            matched_image (Tensor): processed match image data, shape=(sample_num ,512)
            unmatched_image (Tensor): processed unmatch image data, shape=(sample_num, 512)
        """
        text = data['text']
        image = data['image']
        label = data['label']

        # index of batch news data whose label=1
        index = [i for i, l in enumerate(label) if l == 1]
        sample_text = text[index]
        sample_image = image[index]
        fixed_text = copy.deepcopy(sample_text)
        matched_image = copy.deepcopy(sample_image)
        unmatched_image = copy.deepcopy(sample_image).roll(shifts=3, dims=0)
        return fixed_text, matched_image, unmatched_image

    def predict(self, data: Dict[str, Tensor]) -> Tensor:
        """
        predict the probability of being fake news
        Args:
            data Tuple[Tensor]: batch data, including text, image, label
        Returns:
            Tensor: softmax probability, shape=(, 2)
        """
        text = data['text']
        image = data['image']
        output = self.forward(text, image)
        return output


class _GaussianSample(nn.Module):
    def __init__(self, z_dim=2):
        """
        Args:
            z_dim: number of feature dim
        """
        super(_GaussianSample, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): batch data, shape=(batch_size,64)
        Returns:
            Independent: distribution of text or image
        """
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class _AmbiguityModule(nn.Module):
    """
    Cross modal ambiguity learning moddule,
    capture the ambiguity between text and image via KL divergence
    """
    def __init__(self):
        super(_AmbiguityModule, self).__init__()
        self.encoder_text = _GaussianSample()
        self.encoder_image = _GaussianSample()

    def forward(self, text_encoding: Tensor, image_encoding: Tensor) -> Tensor:
        """
        Args:
            text_encoding (Tensor): the batch aligned text, shape=(batch_size, 64)
            image_encoding (Tensor): the batch aligned image, shape=(batch_size, 64)
        Returns:
            Tensor: the ambiguity of text and image, shape=(batch_size, 1)
        """
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.
        skl = nn.functional.sigmoid(skl)
        return skl


class _UnimodalModule(nn.Module):
    """
    linear transform for unimodal data
    """
    def __init__(self, shared_dim=128, prime_dim=16):
        """
        Args:
            shared_dim (int): input dim of data
            prime_dim (int): output dim of data
        """
        super(_UnimodalModule, self).__init__()
        self.text_uni = nn.Sequential(nn.Linear(shared_dim, shared_dim),
                                      nn.BatchNorm1d(shared_dim), nn.ReLU(),
                                      nn.Linear(shared_dim, prime_dim),
                                      nn.BatchNorm1d(prime_dim), nn.ReLU())
        self.image_uni = nn.Sequential(nn.Linear(shared_dim, shared_dim),
                                       nn.BatchNorm1d(shared_dim), nn.ReLU(),
                                       nn.Linear(shared_dim, prime_dim),
                                       nn.BatchNorm1d(prime_dim), nn.ReLU())

    def forward(self, text_encoding: Tensor, image_encoding: Tensor):
        """
        Args:
            text_encoding (Tensor): encoding batch text data, shape=(batch_size, 128)
            image_encoding (Tensor): encoding batch image data, shape=(batch_size, 128)
        Returns:
            text_prime (Tensor): processed batch text data, shape=(batch_size, prime_dim)
            image_prime (Tensor): processed batch image data, shape=(batch_size, prime_dim)
        """
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class _CrossModalModule(nn.Module):
    """
    attention for cross modal fusion
    """
    def __init__(self, corre_out_dim=64):
        """
        Args:
            corre_out_dim (int): output dim of correlation, default=64
        """
        super(_CrossModalModule, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim), nn.ReLU())

    def forward(self, text: Tensor, image: Tensor):
        """
        Args:
            text (Tensor): batch text data, shape=(batch_size, 64)
            image (Tensor): batch image data, shape=(batch_size, 64)
        Returns:
            Tensor: the correlation of text and image, shape=(batch_size, 64)
        """
        text_in = text.unsqueeze(2)
        image_in = image.unsqueeze(1)
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.fc(correlation_p)
        return correlation_out
