# FaKnow

**FaKnow** (**Fa**ke **Know**), a unified *Fake News Detection* algorithms library, including models and training process.

## Features

## Install

## Integrated Models

| category        | paper                                                                                                                                                        | journal/conference | publish year | source repository                                                                                                                   | our code                                                           |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Content Based   | [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)                                                              | EMNLP              | 2014         | [yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)                                                                     | [TextCNN](./template/model/content_based/textcnn.py)               |
|                 | [MDFEND: Multi-domain Fake News Detection](https://dl.acm.org/doi/10.1145/3459637.3482139)                                                                   | CIKM               | 2021         | [kennqiang/MDFEND-Weibo21](https://github.com/kennqiang/MDFEND-Weibo21)                                                             | [MDFEND](./template/model/content_based/mdfend.py)                 |
|                 | [EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3219819.3219903)                            | KDD                | 2018         | [yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18)                                                                   | [EANN](./template/model/content_based/multi_modal/eann.py)         |
|                 | [MFAN: Multi-modal Feature-enhanced Attention Networks for Rumor Detection](https://www.ijcai.org/proceedings/2022/335)                                      | IJCAI              | 2022         | [drivsaf/MFAN](https://github.com/drivsaf/MFAN)                                                                                     | [MFAN](./template/model/content_based/multi_modal/mfan.py)         |
|                 | [SAFE: Similarity-Aware Multi-Modal Fake News Detection](https://dl.acm.org/doi/abs/10.1007/978-3-030-47436-2_27)                                            | PAKDD              | 2020         | [Jindi0/SAFE](https://github.com/Jindi0/SAFE)                                                                                       | [SAFE](./template/model/content_based/multi_modal/safe.py)         |
|                 | [SpotFake: A Multimodal Framework for Fake News Detection](https://ieeexplore.ieee.org/document/8919302)                                                     | IEEE BigMM         | 2019         | [shiivangii/SpotFake](https://github.com/shiivangii/SpotFake)                                                                       | [SpotFake](./template/model/content_based/multi_modal/spotfake.py) |
| Social Context  | [Semi-Supervised Classification with Graph Convolutional Networks](https://ieeexplore.ieee.org/document/8953909)                                             | IEEE CVPR          | 2019         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [Base_GNN](./template/model/social_context/base_gnn.py)            |
|                 | [Inductive Representation Learning on Large Graphs](https://dl.acm.org/doi/10.5555/3294771.3294869)                                                          | NIPS               | 2017         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [Base_GNN](./template/model/social_context/base_gnn.py)            |
|                 | [Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ)                                                                                        | ICLR               | 2018         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [Base_GNN](./template/model/social_context/base_gnn.py)            |
|                 | [Rumor detection on social media with bi-directional graph convolutional networks](https://ojs.aaai.org/index.php/AAAI/article/view/5393)                    | AAAI               | 2020         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [BIGCN](./template/model/social_context/bigcn.py)                  |
|                 | [Fake News Detection on Social Media using Geometric Deep Learning](https://arxiv.org/abs/1902.06673)                                                        | arXiv              | 2019         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [GCNFN](./template/model/social_context/gcnfn.py)                  |
|                 | [Graph neural networks with continual learning for fake news detection from social media](https://arxiv.org/abs/2007.03316)                                  | arXiv              | 2020         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [GNNCL](./template/model/social_context/gnncl.py)                  |
|                 | [User Preference-aware Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3404835.3462990)                                                              | SIGIR              | 2021         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [UPFD](./template/model/social_context/upfd.py)                    |
|                 | [Embracing Domain Differences in Fake News: Cross-domain Fake News Detection using Multi-modal Data](https://ojs.aaai.org/index.php/AAAI/article/view/16134) | AAAI               | 2021         | [amilasilva92/cross-domain-fake-news-detection-aaai2021](https://github.com/amilasilva92/cross-domain-fake-news-detection-aaai2021) | [EDDFN](./template/model/social_context/eddfn.py)                  |
| Knowledge Aware | [Towards Fine-Grained Reasoning for Fake News Detection ](https://aaai.org/papers/05746-towards-fine-grained-reasoning-for-fake-news-detection/)             | AAAI               | 2022         | [Ahren09/FinerFact ](https://github.com/Ahren09/FinerFact)                                                                          | [FinerFact](./template/model/knowledge_aware/finerfact.py)         |

