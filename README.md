# FaKnow

**FaKnow** (**Fa**ke **Know**), a unified *Fake News Detection* algorithms library based on PyTorch, is designed for
reproducing and developing fake news detection algorithms. It includes **17 models**(see at [Integrated Models](##
Integrated Models)), covering **3 categories**:

- content based
- social context
- knowledge aware

## Features

### Unified Framework

*FaKnow* provides a unified and standardised interface to cover a series of algorithm development processes, including
data processing, model writing, training and evaluation, and final result storage. We have designed and developed the
data module, model module and training module as the core components of the framework, and encapsulated many of the
functional components and functions commonly used in fake news detection algorithms.

### Generic Data Structure

For different types of tasks and scenarios, the framework contains different data formats to cope with various needs,
focusing on two main areas of work. At the level of raw data files, as most of the current datasets for disinformation
detection are crawled from social platform pages such as Weibo, we use json as the file format read into the framework
to fit the format of the data crawled down, allowing the user to feed the data into the framework with only minor
processing. At the level of data interaction within the algorithm's internal code, we have developed a uniform
interaction format that allows users to easily access data by referencing feature names.

### Diverse Models

The framework contains a number of representative disinformation detection algorithms published in top academic
conferences or journals, including a variety of content-based, social context-based and knowledge aware detection
algorithm models, giving researchers a wide range of options for replicating their results or using them as a baseline
for their own algorithm research. The built-in models include more than just the classical algorithms, but also focus on
new and popular algorithms from recent years, covering a wide range of perspectives such as text content, multimodality
and information propagation.

### Convenient Usability

The framework is developed based on the *pytorch* and encapsulated in several ways, eliminating the tedious work of
daily model training and evaluation, providing auxiliary functions such as result visualisation, log printing, parameter
saving, etc., avoiding a lot of code redundancy, and also supporting the reading of hyperparameters from configuration
files and command lines for model training. In addition, although the framework has its own wrapper classes and
functions, the overall logic is still the same behaviour commonly used by *pytorch*, and the learning cost is low enough
for any researcher familiar with *pytorch* to start quickly.

### Great Scalability

The classes related to datasets, models, training and evaluation of the framework are designed to shield the internal
detailed code logic and provide call interfaces externally, making the entire framework highly extensible. In addition
to running the built-in algorithms already in the framework, users who want to fine-tune existing algorithms or write
new models of fake news detection algorithms can simply focus on the api exposed by these classes and inherit them
according to the specification to reuse most of the functionality and rewrite a small amount of code to meet their
needs.

## Installation

- from conda

```bash
conda install -c npurg faknow
```

- from pip

```bash
pip install faknow
```

- from source

```bash
git clone https://github.com/NPURG/FaKnow.git && cd FaKnow
pip install -e . --verbose
```

## Quick Start

We provide several methods to **run integrated models** quickly with passing only few arguments. For hyper parameters
like learning rate, values from the open source code of the paper are taken as default. You can also passing your own
defined hyper parameters to these methods.

### run

You can use `run` and `run_from_yaml` methods to run integrated models. The former receives the parameters as `dict`
keyword arguments and the latter reads them from the `yaml` configuration file.

- run from kargs

```python
from faknow.run import run

model = 'mdfend'  # lowercase short name of models
kargs = {'train_path': 'train.json', 'test_path': 'test.json'}  # dict arguments
run(model, **kargs)
```

- run from yaml

```python
# demo.py
from faknow.run import run_from_yaml

model = 'mdfend'  # lowercase short name of models
config_path = 'mdfend.yaml'  # config file path
run_from_yaml(model, config_path)
```

your yaml config file should be like:

```yaml
# mdfend.yaml
train_path: train.json # the path of training set file
test_path: test.json # the path of testing set file
```

### run specific models

You can also run specific models using `run_model` and `run_model_from_yaml` methods by passing paramter, where `model`
is the short name of the integrated model you want to use. The usages are the same as `run` and `run_from_yaml`.
Following is an example to run *mdfend*.

```python
from faknow.run.content_based.run_mdfend import run_mdfend, run_mdfend_from_yaml

# run from kargs
kargs = {'train_path': 'train.json', 'test_path': 'test.json'}  # dict training arguments
run_mdfend(**kargs)

# or run from yaml
config_path = 'mdfend.yaml'  # config file path
run_mdfend_from_yaml(config_path)
```

## Integrated Models

| category        | paper                                                                                                                                                        | journal/conference | publish year | source repository                                                                                                                   | our code                                                       |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| Content Based   | [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)                                                              | EMNLP              | 2014         | [yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)                                                                     | [TextCNN](faknow/model/content_based/textcnn.py)               |
|                 | [MDFEND: Multi-domain Fake News Detection](https://dl.acm.org/doi/10.1145/3459637.3482139)                                                                   | CIKM               | 2021         | [kennqiang/MDFEND-Weibo21](https://github.com/kennqiang/MDFEND-Weibo21)                                                             | [MDFEND](faknow/model/content_based/mdfend.py)                 |
|                 | [EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3219819.3219903)                            | KDD                | 2018         | [yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18)                                                                   | [EANN](faknow/model/content_based/multi_modal/eann.py)         |
|                 | [MFAN: Multi-modal Feature-enhanced Attention Networks for Rumor Detection](https://www.ijcai.org/proceedings/2022/335)                                      | IJCAI              | 2022         | [drivsaf/MFAN](https://github.com/drivsaf/MFAN)                                                                                     | [MFAN](faknow/model/content_based/multi_modal/mfan.py)         |
|                 | [SAFE: Similarity-Aware Multi-Modal Fake News Detection](https://dl.acm.org/doi/abs/10.1007/978-3-030-47436-2_27)                                            | PAKDD              | 2020         | [Jindi0/SAFE](https://github.com/Jindi0/SAFE)                                                                                       | [SAFE](faknow/model/content_based/multi_modal/safe.py)         |
|                 | [SpotFake: A Multimodal Framework for Fake News Detection](https://ieeexplore.ieee.org/document/8919302)                                                     | BigMM              | 2019         | [shiivangii/SpotFake](https://github.com/shiivangii/SpotFake)                                                                       | [SpotFake](faknow/model/content_based/multi_modal/spotfake.py) |
|                 | [Multimodal Fusion with Co-Attention Networks for Fake News Detection](https://aclanthology.org/2021.findings-acl.226/)                                      | ACL                | 2021         | [wuyang45/MCAN_code](https://github.com/wuyang45/MCAN_code)                                                                         | [MCAN](faknow/model/content_based/multi_modal/mcan.py)         |
| Social Context  | [Semi-Supervised Classification with Graph Convolutional Networks](https://ieeexplore.ieee.org/document/8953909)                                             | CVPR               | 2019         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [GCN](faknow/model/social_context/base_gnn.py)                 |
|                 | [Inductive Representation Learning on Large Graphs](https://dl.acm.org/doi/10.5555/3294771.3294869)                                                          | NeurIPS            | 2017         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [SAGE](faknow/model/social_context/base_gnn.py)                |
|                 | [Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ)                                                                                        | ICLR               | 2018         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [GAT](faknow/model/social_context/base_gnn.py)                 |
|                 | [Rumor detection on social media with bi-directional graph convolutional networks](https://ojs.aaai.org/index.php/AAAI/article/view/5393)                    | AAAI               | 2020         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [BIGCN](faknow/model/social_context/bigcn.py)                  |
|                 | [Fake News Detection on Social Media using Geometric Deep Learning](https://arxiv.org/abs/1902.06673)                                                        | arXiv              | 2019         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [GCNFN](faknow/model/social_context/gcnfn.py)                  |
|                 | [Graph neural networks with continual learning for fake news detection from social media](https://arxiv.org/abs/2007.03316)                                  | arXiv              | 2020         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [GNNCL](faknow/model/social_context/gnncl.py)                  |
|                 | [User Preference-aware Fake News Detection](https://dl.acm.org/doi/abs/10.1145/3404835.3462990)                                                              | SIGIR              | 2021         | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)                                                               | [UPFD](faknow/model/social_context/upfd.py)                    |
|                 | [Embracing Domain Differences in Fake News: Cross-domain Fake News Detection using Multi-modal Data](https://ojs.aaai.org/index.php/AAAI/article/view/16134) | AAAI               | 2021         | [amilasilva92/cross-domain-fake-news-detection-aaai2021](https://github.com/amilasilva92/cross-domain-fake-news-detection-aaai2021) | [EDDFN](faknow/model/social_context/eddfn.py)                  |
|                 | [Zoom Out and Observe: News Environment Perception for Fake News Detection](https://aclanthology.org/2022.acl-long.311/)                                     | ACL                | 2022         | [ictmcg/news-environment-perception](https://github.com/ICTMCG/News-Environment-Perception)                                         | [NEP](faknow/model/social_context/nep.py)                      |
| Knowledge Aware | [Towards Fine-Grained Reasoning for Fake News Detection ](https://aaai.org/papers/05746-towards-fine-grained-reasoning-for-fake-news-detection/)             | AAAI               | 2022         | [Ahren09/FinerFact ](https://github.com/Ahren09/FinerFact)                                                                          | [FinerFact](faknow/model/knowledge_aware/finerfact.py)         |

## Citation

```tex

```

## License

FaKnow has a **MIT**-style license, as found in the [LICENSE](./LICENSE) file.
