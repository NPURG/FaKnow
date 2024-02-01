# Introduction

<p align="center">
    <a href="https://faknow.readthedocs.io">
        <img alt="doc" src="https://img.shields.io/badge/doc-read_the_docs-blue.svg">
    </a>
    <a href="https://github.com//NPURG/FaKnow/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/license-MIT-green.svg">
    </a>
    <a href="https://github.com/NPURG/FaKnow/releases">
        <img alt="release" src="https://img.shields.io/badge/relase-v0.0.3-yellow.svg">
    </a>
    <!-- <a href="https://github.com/huggingface/transformers/releases">
        <img alt="doi" src="https://img.shields.io/badge/doi-xxx-orange.svg"> -->
    </a>
</p>


**FaKnow** (**Fa**ke **Know**), a unified *Fake News Detection* algorithms library based on PyTorch, is designed for
reproducing and developing fake news detection algorithms. It includes **22 models**(see at **Integrated Models**), covering **2 categories**:

- content based
- social context

## Integrated Models

| category       | paper                                                                                                                                                             | journal/conference | publish year | source repository                                                                     | our code                                                       |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------|---------------------------------------------------------------------------------------|----------------------------------------------------------------|
| Content Based  | [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)                                                                   | EMNLP              | 2014         | [yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)                       | [TextCNN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/textcnn.py)               |
|                | [EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3219819.3219903)                                 | KDD                | 2018         | [yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18)                     | [EANN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/multi_modal/eann.py)         |
|                | [SpotFake: A Multimodal Framework for Fake News Detection](https://ieeexplore.ieee.org/document/8919302)                                                          | BigMM              | 2019         | [shiivangii/SpotFake](https://github.com/shiivangii/SpotFake)                         | [SpotFake](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/multi_modal/spotfake.py) |
|                | [SAFE: Similarity-Aware Multi-Modal Fake News Detection](https://dl.acm.org/doi/abs/10.1007/978-3-030-47436-2_27)                                                 | PAKDD              | 2020         | [Jindi0/SAFE](https://github.com/Jindi0/SAFE)                                         | [SAFE](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/multi_modal/safe.py)         |
|                | [MDFEND: Multi-domain Fake News Detection](https://dl.acm.org/doi/10.1145/3459637.3482139)                                                                        | CIKM               | 2021         | [kennqiang/MDFEND-Weibo21](https://github.com/kennqiang/MDFEND-Weibo21)               | [MDFEND](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/mdfend.py)                 |
|                | [Multimodal Fusion with Co-Attention Networks for Fake News Detection](https://aclanthology.org/2021.findings-acl.226/)                                           | ACL                | 2021         | [wuyang45/MCAN_code](https://github.com/wuyang45/MCAN_code)                           | [MCAN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/multi_modal/mcan.py)         |
|                | [HMCAN: Hierarchical Multi-modal Contextual Attention Network for fake news Detection](https://dl.acm.org/doi/10.1145/3404835.3462871)                            | SIGIR              | 2021         | [wangjinguang502/HMCAN](https://github.com/wangjinguang502/HMCAN)                     | [HMCAN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/multi_modal/hmcan.py)       |
|                | [MFAN: Multi-modal Feature-enhanced Attention Networks for Rumor Detection](https://www.ijcai.org/proceedings/2022/335)                                           | IJCAI              | 2022         | [drivsaf/MFAN](https://github.com/drivsaf/MFAN)                                       | [MFAN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/multi_modal/mfan.py)         |
|                | [Generalizing to the Future: Mitigating Entity Bias in Fake News Detection](https://dl.acm.org/doi/10.1145/3477495.3531816)                                       | SIGIR              | 2022         | [ICTMCG/ENDEF-SIGIR2022](https://github.com/ICTMCG/ENDEF-SIGIR2022)                   | [ENDEF](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/endef.py)                   |
|                | [M3FEND: Memory-Guided Multi-View Multi-Domain Fake News Detection](https://ieeexplore.ieee.org/document/9802916)                                                 | TKDE               | 2022         | [ICTMCG/M3FEND](https://github.com/ICTMCG/M3FEND)                                     | [M3FEND](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/m3fend.py)                 |
|                | [CAFE: Cross-modal Ambiguity Learning for Multimodal Fake News Detection](https://dl.acm.org/doi/10.1145/3485447.3511968)                                         | WWW                | 2022         | [cyxanna/CAFE](https://github.com/cyxanna/CAFE)                                       | [CAFE](https://github.com/NPURG/FaKnow/blob/master/faknow/model/content_based/multi_modal/cafe.py)         |
| Social Context | [Semi-Supervised Classification with Graph Convolutional Networks](https://ieeexplore.ieee.org/document/8953909)                                                  | ICLR               | 2017         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                 | [GCN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/base_gnn.py)                 |
|                | [Inductive Representation Learning on Large Graphs](https://dl.acm.org/doi/10.5555/3294771.3294869)                                                               | NeurIPS            | 2017         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                 | [GraphSAGE](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/base_gnn.py)           |
|                | [Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ)                                                                                             | ICLR               | 2018         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                 | [GAT](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/base_gnn.py)                 |
|                | [Fake News Detection on Social Media using Geometric Deep Learning](https://arxiv.org/abs/1902.06673)                                                             | arXiv              | 2019         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                 | [GCNFN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/gcnfn.py)                  |
|                | [Rumor detection on social media with bi-directional graph convolutional networks](https://ojs.aaai.org/index.php/AAAI/article/view/5393)                         | AAAI               | 2020         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                 | [BIGCN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/bigcn.py)                  |
|                | [FANG: Leveraging Social Context for Fake News Detection Using Graph Representation](https://dl.acm.org/doi/10.1145/3340531.3412046)                              | CIKM               | 2020         | [nguyenvanhoang7398/FANG](https://github.com/nguyenvanhoang7398/FANG)                 | [Fang](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/fang.py)                    |
|                | [Graph neural networks with continual learning for fake news detection from social media](https://arxiv.org/abs/2007.03316)                                       | arXiv              | 2020         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                 | [GNNCL](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/gnncl.py)                  |
|                | [User Preference-aware Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3404835.3462990)                                                                   | SIGIR              | 2021         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                 | [UPFD](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/upfd.py)                    |
|                | [DUDEF: Mining Dual Emotion for Fake News Detection](https://dl.acm.org/doi/10.1145/3442381.3450004)                                                              | WWW                | 2021         | [RMSnow/WWW2021](https://github.com/RMSnow/WWW2021)                                   | [DUDEF](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/dudef.py)                  |
|                | [Towards Propagation Uncertainty: Edge-enhanced Bayesian Graph Convolutional Networks for Rumor Detection, ACL 2021](https://aclanthology.org/2021.acl-long.297/) | ACL                | 2021         | [weilingwei96/EBGCN](https://github.com/weilingwei96/EBGCN)                           | [EBGCN](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/ebgcn.py)                  |
|                | [Towards Trustworthy Rumor Detection with Interpretable Graph Structural Learning](https://dl.acm.org/doi/10.1145/3583780.3615228)                                | CIKM               | 2023         | [Anonymous4ScienceAuthor/TrustRD](https://github.com/Anonymous4ScienceAuthor/TrustRD) | [TrustRD](https://github.com/NPURG/FaKnow/blob/master/faknow/model/social_context/trustrd.py)              |



## Features

- **Unified Framework**: provide a unified interface to cover a series of algorithm development processes, including data processing, model developing, training and evaluation
- **Generic Data Structure**: use json as the file format read into the framework to fit the format of the data crawled down, allowing the user to customize the processing of different fields
- **Diverse Models**: contains a number of representative fake news detection algorithms published in conferences or journals during recent years, including a variety of content-based and social context-based models
- **Convenient Usability**: pytorch based style makes it easy to use with rich auxiliary functions like loss visualization, logging, parameter saving
- **Great Scalability**: users just focus on the exposed API and inherit built-in classes to reuse most of the functionality and only need to write a little code to meet new requirements

## Citation

```tex
@misc{faknow,
  title = {{{FaKnow}}: {{A Unified Library}} for {{Fake News Detection}}},
  shorttitle = {{{FaKnow}}},
  author = {Zhu, Yiyuan and Li, Yongjun and Wang, Jialiang and Gao, Ming and Wei, Jiali},
  year = {2024},
  month = jan,
  number = {arXiv:2401.16441},
  eprint = {2401.16441},
  primaryclass = {cs},
  publisher = {{arXiv}},
  archiveprefix = {arxiv},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Computation and Language,Computer Science - Machine Learning}
}
```

## License

FaKnow has a **MIT**-style license, as found in the [LICENSE](./LICENSE) file.