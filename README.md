# FaKnow

**FaKnow** (**Fa**ke **Know**), a unified *Fake News Detection* algorithms library, including models and training process.

## Features


## Install



## Integrated Models

| category        | paper                                                                                                                                            | journal/conference | publish year | source repository                                                       | our code                                                   |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------|-------------------------------------------------------------------------|------------------------------------------------------------|
| Content Based   | [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)                                                  | EMNLP              | 2014         | [yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)         | [TextCNN](./template/model/content_based/textcnn.py)       |
|                 | [MDFEND: Multi-domain Fake News Detection](https://dl.acm.org/doi/10.1145/3459637.3482139)                                                       | CIKM               | 2021         | [kennqiang/MDFEND-Weibo21](https://github.com/kennqiang/MDFEND-Weibo21) | [MDFEND](./template/model/content_based/mdfend.py)         |
| Social Context  | [User Preference-aware Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3404835.3462990)                                                  | SIGIR              | 2021         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)   | [UPFD](./template/model/social_context/upfd.py)            |
| Knowledge Aware | [Towards Fine-Grained Reasoning for Fake News Detection ](https://aaai.org/papers/05746-towards-fine-grained-reasoning-for-fake-news-detection/) | AAAI               | 2022         | [Ahren09/FinerFact ](https://github.com/Ahren09/FinerFact)              | [FinerFact](./template/model/knowledge_aware/finerfact.py) |

