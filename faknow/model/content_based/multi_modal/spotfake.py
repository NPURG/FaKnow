import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from transformers import BertModel

from faknow.model.model import AbstractModel

device = torch.device("cuda")
"""
SpotFake: Multi-Modal Fake News Detection
paper: https://ieeexplore.ieee.org/document/8919302
code: https://github.com/shiivangii/SpotFake
"""


# 文本Bert基本模型
class _TextEncoder(nn.Module):
    def __init__(
            self,
            text_fc2_out=32,
            text_fc1_out=2742,
            dropout_p=0.4,
            fine_tune_module=False,
            pre_trained_bert_name="bert-base-uncased",
    ):
        super(_TextEncoder, self).__init__()
        self.fine_tune_module = fine_tune_module
        # 实例化
        self.bert = BertModel.from_pretrained(pre_trained_bert_name).requires_grad_(False)
        self.embedding_size = self.bert.config.hidden_size
        self.text_enc_fc1 = torch.nn.Linear(self.embedding_size, text_fc1_out)
        self.text_enc_fc2 = torch.nn.Linear(text_fc1_out, text_fc2_out)
        self.dropout = nn.Dropout(dropout_p)
        self.fine_tune()

    def forward(self, input_ids, attention_mask):
        """
        输入Bert和分类器，计算logis
        @参数    input_ids (torch.Tensor): 输入 (batch_size,max_length)
        @参数    attention_mask (torch.Tensor): attention mask information (batch_size, max_length)
        @返回    logits (torch.Tensor): 输出 (batch_size, num_labels)
        """

        # 输入BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc1(out['pooler_output']))
        )
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc2(x))
        )
        return x

    def fine_tune(self):
        """
        固定参数
        """
        for p in self.bert.parameters():
            p.requires_grad = self.fine_tune_module


# 视觉vgg19预训练模型
class _VisionEncoder(nn.Module):
    def __init__(
            self,
            img_fc1_out=2742,
            img_fc2_out=32,
            dropout_p=0.4,
            fine_tune_module=False
    ):
        super(_VisionEncoder, self).__init__()
        self.fine_tune_module = fine_tune_module
        # 实例化
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        self.vis_encoder = vgg
        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)
        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)
        self.dropout = nn.Dropout(dropout_p)
        self.fine_tune()

    def forward(self, images):
        """
        :参数: images, tensor (batch_size, 3, image_size, image_size)
        :返回: encoded images
        """

        x = self.vis_encoder(images)
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )
        return x

    def fine_tune(self):
        """
        允许或阻止vgg的卷积块2到4的梯度计算。
        """
        for p in self.vis_encoder.parameters():
            p.requires_grad = False
        # 如果进行微调，则只微调卷积块2到4
        for c in list(self.vis_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module


# LanguageAndVisionConcat
class _TextConcatVision(nn.Module):
    def __init__(
            self,
            model_params
    ):
        super(_TextConcatVision, self).__init__()
        self.text_encoder = _TextEncoder(model_params['text_fc2_out'], model_params['text_fc1_out'],
                                         model_params['dropout_p'], model_params['fine_tune_text_module'],
                                         model_params['pre_trained_bert_name'])
        self.vision_encoder = _VisionEncoder(model_params['img_fc1_out'], model_params['img_fc2_out'],
                                             model_params['dropout_p'], model_params['fine_tune_vis_module'])
        self.fusion = torch.nn.Linear(
            in_features=(model_params['text_fc2_out'] + model_params['img_fc2_out']),
            out_features=model_params['fusion_output_size']
        )
        self.fc = torch.nn.Linear(
            in_features=model_params['fusion_output_size'],
            out_features=1
        )
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])

    # def forward(self, text, image):
    def forward(self, text, image):
        # text to Bert
        text_features = self.text_encoder(text[0], text[1])
        # image to vgg
        image_features = self.vision_encoder(image)
        # 连接image & text
        combined_features = torch.cat(
            [text_features, image_features], dim=1
        )
        combined_features = self.dropout(combined_features)
        fused = self.dropout(
            torch.relu(
                self.fusion(combined_features)
            )
        )

        prediction = torch.sigmoid(self.fc(fused))
        prediction = prediction.squeeze(-1)
        prediction = prediction.float()
        return prediction


class SpotFake(AbstractModel):
    def __init__(
            self,
            text_fc2_out: int = 32,
            text_fc1_out: int = 2742,
            dropout_p: float = 0.4,
            fine_tune_text_module: bool = False,
            img_fc1_out: int = 2742,
            img_fc2_out: int = 32,
            fine_tune_vis_module: bool = False,
            fusion_output_size: int = 35,
            loss_func=nn.BCELoss(),
            pre_trained_bert_name="bert-base-uncased"
    ):
        super(SpotFake, self).__init__()
        self.text_fc2_out = text_fc2_out
        self.text_fc1_out = text_fc1_out
        self.dropout_p = dropout_p
        self.fine_tune_text_module = fine_tune_text_module
        self.img_fc1_out = img_fc1_out
        self.img_fc2_out = img_fc2_out
        self.fine_tune_vis_module = fine_tune_vis_module
        self.fusion_output_size = fusion_output_size
        self.pre_trained_bert_name = pre_trained_bert_name
        model_params = {
            "text_fc2_out": text_fc2_out,
            "text_fc1_out": text_fc1_out,
            "dropout_p": dropout_p,
            "fine_tune_text_module": fine_tune_text_module,
            "img_fc1_out": img_fc1_out,
            "img_fc2_out": img_fc2_out,
            "fine_tune_vis_module": fine_tune_vis_module,
            "fusion_output_size": fusion_output_size,
            "pre_trained_bert_name": pre_trained_bert_name
        }
        self.model = _TextConcatVision(model_params).to(device)
        if loss_func is None:
            self.loss_func = nn.BCELoss()
        else:
            self.loss_func = loss_func

    def forward(self, text: torch.Tensor, mask: torch.Tensor, domain: torch.Tensor):
        return self.model([text, mask], image=domain)

    def calculate_loss(self, data) -> Tensor:
        img_ip, text_ip, label = data["image_id"], data["BERT_ip"], data['label']
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)
        imgs_ip = img_ip.to(device)
        b_labels = label.to(device)
        output = self.forward(b_input_ids, b_attn_mask, imgs_ip)
        return self.loss_func(output, b_labels.float())

    @torch.no_grad()
    def predict(self, data_without_label):
        img_ip, text_ip = data_without_label["image_id"], data_without_label["BERT_ip"]
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)
        imgs_ip = img_ip.to(device)

        # shape=(n,), data = 1 or 0
        round_pred = self.forward(b_input_ids, b_attn_mask, imgs_ip)

        new_outputs = torch.zeros((round_pred.shape[0], 2)).to(round_pred.device)

        new_outputs[torch.where(round_pred < 0.5)[0].detach().cpu().numpy(), 0] = 1 - round_pred[round_pred < 0.5]
        new_outputs[torch.where(round_pred < 0.5)[0].detach().cpu().numpy(), 1] = round_pred[round_pred < 0.5]

        new_outputs[torch.where(round_pred >= 0.5)[0].detach().cpu().numpy(), 1] = round_pred[round_pred >= 0.5]
        new_outputs[torch.where(round_pred >= 0.5)[0].detach().cpu().numpy(), 0] = 1 - round_pred[round_pred >= 0.5]

        return new_outputs
