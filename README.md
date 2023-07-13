# FaKnow

**FaKnow** (**Fa**ke **Know**), a unified *Fake News Detection* algorithms library based on PyTorch, is designed for
reproducing and developing fake news detection algorithms. It includes **17 models**(see at **Integrated Models**), covering **3 categories**:

- content based
- social context
- knowledge aware



## Features

- **Unified Framework**: provide a unified and standardised interface to cover a series of algorithm development processes, including data processing, model developing, training and evaluation
- **Generic Data Structure**:  use json as the file format read into the framework to fit the format of the data crawled down, allowing the user to feed the data into the framework with only minor processing

- **Diverse Models**: contains a number of representative fake news detection algorithms published in conferences or journals during recent years, including a variety of content-based, social context-based and knowledge aware models
- **Convenient Usability**: pytorch based style make it easy to use with rich auxiliary functions like result visualisation, log printing, parameter saving

- **Great Scalability**: just focus on the exposed api and inherit built-in classed to reuse most of the functionality and only need to write a little code to meet new requirements



## Installation

FaKnow is available for **Python 3.8** and higher. 

**Make sure [PyTorch](https://pytorch.org/)(including torch and torchvision) and [PyG](https://www.pyg.org/)(including torch_geometric and optional dependencies) are already installed.**

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



## Usage Examples

### Quick Start

We provide several methods to **run integrated models** quickly with passing only few arguments. For hyper parameters
like learning rate, values from the open source code of the paper are taken as default. You can also passing your own
defined hyper parameters to these methods.

#### run

You can use `run` and `run_from_yaml` methods to run integrated models. The former receives the parameters as `dict`
keyword arguments and the latter reads them from the `yaml` configuration file.

- run from kargs

```python
from faknow.run import run

model = 'mdfend'  # lowercase short name of models
kargs = {'train_path': 'train.json', 'test_path': 'test.json'}  # dict arguments
run(model, **kargs)
```

the json file for *mdfend* shoule be like:

```json
[
    {
        "text": "this is a sentence.",
        "domain": 9
    },
    {
        "text": "this is a sentence.",
        "domain": 1
    }
]
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

#### run specific models

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

### Run From Scratch

Following is an example to run *mdfend* from scratch.

```python
from faknow.data.dataset.text import TextDataset
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.mdfend import MDFEND
from faknow.train.trainer import BaseTrainer
from faknow.run.content_based import TokenizerMDFEND

import torch
from torch.utils.data import DataLoader

# tokenizer for MDFEND
max_len, bert = 170, 'bert-base-uncased'
tokenizer = TokenizerMDFEND(max_len, bert)

batch_size = 64
# dataset path
train_path, test_path, validate_path = 'train.json', 'test.json', 'val.json'
train_set = TextDataset(train_path, ['text'], tokenizer)
train_loader = DataLoader(train_set, batch_size, shuffle=True)

validate_set = TextDataset(validate_path, ['text'], tokenizer)
val_loader = DataLoader(validate_set, batch_size, shuffle=False)

test_set = TextDataset(test_path, ['text'], tokenizer)
tset_loader = DataLoader(test_set, batch_size, shuffle=False)

# prepare model
domain_num = 9
model = MDFEND(bert, domain_num)

# optimizer and lr scheduler
lr, weight_decay, step_size, gamma = 0.00005, 5e-5, 100, 0.98
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

# metrics to evaluate the model performance
evaluator = Evaluator()

# train and validate
num_epochs, device = 50, 'cpu'
trainer = BaseTrainer(model, evaluator, optimizer, scheduler, device=device)
trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

# show test result
print(trainer.evaluate(test_loader))
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
